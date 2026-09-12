// Operator pilot only: this does not yet advance Navier-Stokes or the paper force.
#include <AMReX.H>
#include <AMReX_BoxIterator.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <hydro_MacProjector.H>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numbers>
#include <string>
#include <sys/resource.h>

using namespace amrex;
using FaceField = Array<MultiFab, 3>;
constexpr Real pi = std::numbers::pi_v<Real>;

Real sinc(Real x) { return std::sin(x) / x; }

// Exact face averages, not point samples. This makes the analytic solenoidal
// flux conservative at both same-level and coarse/fine faces.
Real solenoidal(IntVect const& index, int direction, Real dx)
{
    Real value = direction == 2 ? -2.0 : 1.0;
    for (int d = 0; d < 3; ++d) {
        Real x = -1.0 + (index[d] + (d == direction ? 0.0 : 0.5)) * dx;
        value *= d == direction ? std::sin(pi*x)
                                : std::cos(pi*x) * sinc(pi*dx/2);
    }
    return value;
}

Real gradient(IntVect const& index, int direction, Real dx)
{
    // Gradient of 0.05 cos(pi*x) cos(2*pi*y) cos(3*pi*z).
    // Distinct wave numbers expose spatial truncation error even on one level.
    Real value = -0.05 * (direction+1) * pi;
    for (int d = 0; d < 3; ++d) {
        Real k = (d+1)*pi;
        Real x = -1.0 + (index[d] + (d == direction ? 0.0 : 0.5)) * dx;
        value *= d == direction ? std::sin(k*x)
                                : std::cos(k*x) * sinc(k*dx/2);
    }
    return value;
}

struct Statistics {
    Real error2 = 0, reference2 = 0, div2 = 0, div_inf = 0;
    Real velocity_inf = 0, error_inf = 0, interface_error_inf = 0, volume = 0;
    Long active_cells = 0;
};

Statistics measure(Vector<FaceField> const& velocity, Vector<FaceField> const& exact,
                   Vector<Geometry> const& geometry, Vector<BoxArray> const& grids,
                   Vector<DistributionMapping> const& mapping)
{
    Statistics s;
    int levels = static_cast<int>(geometry.size());
    for (int lev = 0; lev < levels; ++lev) {
        Real dx = geometry[lev].CellSize(0), dv = dx*dx*dx;
        MultiFab divergence(grids[lev], mapping[lev], 1, 0);
        computeDivergence(divergence, GetArrOfConstPtrs(velocity[lev]), geometry[lev]);
        iMultiFab mask(grids[lev], mapping[lev], 1, 0);
        mask.setVal(1);
        if (lev+1 < levels) {
            mask = makeFineMask(grids[lev], mapping[lev], grids[lev+1], IntVect(2), 1, 0);
        }
        for (MFIter mfi(divergence); mfi.isValid(); ++mfi) {
            auto div = divergence.const_array(mfi);
            auto valid = mask.const_array(mfi);
            for (BoxIterator it(mfi.validbox()); it.ok(); ++it) {
                IntVect const& i = it();
                if (!valid(i)) continue; // Never double-count covered coarse cells.
                ++s.active_cells;
                s.volume += dv;
                s.div2 += div(i)*div(i)*dv;
                s.div_inf = std::max(s.div_inf, std::abs(div(i)));
                bool interface = false;
                for (int boundary = 1; boundary < levels; ++boundary) {
                    Real halfwidth = std::ldexp(1.0, -boundary);
                    bool inside = true, near_face = false;
                    for (int d = 0; d < 3; ++d) {
                        Real x = -1.0 + (i[d]+0.5)*dx;
                        inside = inside && std::abs(x) <= halfwidth+2*dx;
                        near_face = near_face || std::abs(std::abs(x)-halfwidth) <= 2*dx;
                    }
                    interface = interface || (inside && near_face);
                }
                for (int d = 0; d < 3; ++d) {
                    auto u = velocity[lev][d].const_array(mfi);
                    auto v = exact[lev][d].const_array(mfi);
                    IntVect j = i; ++j[d];
                    Real mean = 0.5*(u(i)+u(j));
                    Real target = 0.5*(v(i)+v(j));
                    Real error = mean-target;
                    s.error2 += error*error*dv;
                    s.reference2 += target*target*dv;
                    s.error_inf = std::max(s.error_inf, std::abs(error));
                    s.velocity_inf = std::max({s.velocity_inf, std::abs(u(i)), std::abs(u(j))});
                    if (interface) s.interface_error_inf = std::max(s.interface_error_inf, std::abs(error));
                }
            }
        }
    }
    ParallelDescriptor::ReduceRealSum(s.error2);
    ParallelDescriptor::ReduceRealSum(s.reference2);
    ParallelDescriptor::ReduceRealSum(s.div2);
    ParallelDescriptor::ReduceRealSum(s.volume);
    ParallelDescriptor::ReduceRealMax(s.div_inf);
    ParallelDescriptor::ReduceRealMax(s.error_inf);
    ParallelDescriptor::ReduceRealMax(s.velocity_inf);
    ParallelDescriptor::ReduceRealMax(s.interface_error_inf);
    ParallelDescriptor::ReduceLongSum(s.active_cells);
    return s;
}

Real flux_mismatch(Vector<FaceField> const& velocity, Vector<Geometry> const& geometry)
{
    Real result = 0;
    for (int lev = 1; lev < static_cast<int>(velocity.size()); ++lev) {
        FaceField averaged;
        for (int d = 0; d < 3; ++d) {
            auto const& coarse = velocity[lev-1][d];
            averaged[d].define(coarse.boxArray(), coarse.DistributionMap(), 1, 0);
            MultiFab::Copy(averaged[d], coarse, 0, 0, 1, 0);
        }
        average_down_faces(GetArrOfConstPtrs(velocity[lev]), GetArrOfPtrs(averaged),
                           IntVect(2), geometry[lev-1]);
        for (int d = 0; d < 3; ++d) {
            MultiFab::Subtract(averaged[d], velocity[lev-1][d], 0, 0, 1, 0);
            result = std::max(result, averaged[d].norm0());
        }
    }
    return result;
}

int pilot()
{
    ParmParse pp;
    int n = 16, levels = 3, max_grid = 32;
    std::string test_case = "mixed";
    pp.query("n_cell", n); pp.query("levels", levels);
    pp.query("max_grid_size", max_grid); pp.query("test_case", test_case);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n >= 16 && n <= 128 && n%8 == 0, "pilot n_cell must be a multiple of 8 in [16,128]");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(levels >= 1 && levels <= 4, "pilot supports 1 to 4 levels");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(max_grid >= 8 && max_grid%8 == 0, "max_grid_size must be a positive multiple of 8");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(test_case == "mixed" || test_case == "rest" || test_case == "solenoidal", "unknown test_case");
    Real start = amrex::second();
    Vector<Geometry> geometry;
    Vector<BoxArray> grids;
    Vector<DistributionMapping> mapping;
    Vector<FaceField> velocity(levels), exact(levels), projected_once(levels);
    RealBox physical({-1., -1., -1.}, {1., 1., 1.});
    int periodic[3] = {1, 1, 1};
    Long stored_cells = 0;
    for (int lev = 0; lev < levels; ++lev) {
        int full_n = n*(1<<lev);
        Box domain(IntVect(0), IntVect(full_n-1));
        geometry.emplace_back(domain, &physical, 0, periodic);
        int lower = (full_n-n)/2;
        Box refined(IntVect(lower), IntVect(lower+n-1));
        BoxArray boxes(lev == 0 ? domain : refined);
        boxes.maxSize(max_grid);
        stored_cells += boxes.numPts();
        grids.push_back(boxes);
        mapping.emplace_back(boxes);
        Real dx = geometry[lev].CellSize(0);
        for (int d = 0; d < 3; ++d) {
            BoxArray faces = amrex::convert(boxes, IntVect::TheDimensionVector(d));
            velocity[lev][d].define(faces, mapping[lev], 1, 0);
            exact[lev][d].define(faces, mapping[lev], 1, 0);
            projected_once[lev][d].define(faces, mapping[lev], 1, 0);
            for (MFIter mfi(velocity[lev][d]); mfi.isValid(); ++mfi) {
                auto u = velocity[lev][d].array(mfi), v = exact[lev][d].array(mfi);
                for (BoxIterator it(mfi.validbox()); it.ok(); ++it) {
                    auto const& i = it();
                    v(i) = test_case == "rest" ? 0.0 : solenoidal(i, d, dx);
                    u(i) = v(i) + (test_case == "mixed" ? gradient(i, d, dx) : 0.0);
                }
            }
        }
    }
    Statistics before = measure(velocity, exact, geometry, grids, mapping);
    Hydro::MacProjector projection(GetVecOfArrOfPtrs(velocity), 1.0, geometry, LPInfo());
    Array<LinOpBCType, 3> bc{LinOpBCType::Periodic, LinOpBCType::Periodic, LinOpBCType::Periodic};
    projection.setDomainBC(bc, bc);
    projection.setVerbose(0);
    projection.project(1.e-11, 1.e-12);
    Statistics after = measure(velocity, exact, geometry, grids, mapping);
    Real mismatch = flux_mismatch(velocity, geometry);
    for (int lev = 0; lev < levels; ++lev)
        for (int d = 0; d < 3; ++d)
            MultiFab::Copy(projected_once[lev][d], velocity[lev][d], 0, 0, 1, 0);
    projection.project(1.e-11, 1.e-12);
    Statistics repeated = measure(velocity, projected_once, geometry, grids, mapping);
    struct rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
    Real peak_mib = usage.ru_maxrss / 1048576.0;
#else
    Real peak_mib = usage.ru_maxrss / 1024.0;
#endif
    bool passed = std::isfinite(after.error2) && std::isfinite(after.div_inf)
        && after.div_inf < 1.e-8 && mismatch < 1.e-12
        && std::sqrt(repeated.error2/after.volume) < 1.e-9
        && std::abs(after.volume-8.0) < 1.e-10
        && (test_case != "solenoidal" || std::sqrt(after.error2/after.volume) < 1.e-9)
        && (test_case != "rest" || after.velocity_inf == 0.0);
    if (ParallelDescriptor::IOProcessor()) {
        amrex::Print() << std::setprecision(17)
          << "NS_PILOT_RESULT {\"schema_version\":1,\"backend\":\"AMReX-Hydro MAC/MLMG\","
          << "\"scope\":\"static-mesh projection pilot; not a Navier-Stokes run\","
          << "\"pilot_source_sha256\":\"" << NS_PILOT_SOURCE_SHA256 << "\","
          << "\"amrex_commit\":\"" << NS_AMREX_COMMIT << "\",\"hydro_commit\":\"" << NS_HYDRO_COMMIT << "\","
          << "\"test_case\":\"" << test_case << "\",\"base_n\":" << n << ",\"levels\":" << levels
          << ",\"max_grid_size\":" << max_grid << ",\"finest_effective_n\":" << n*(1<<(levels-1))
          << ",\"stored_cells\":" << stored_cells << ",\"active_cells\":" << after.active_cells
          << ",\"composite_volume\":" << after.volume
          << ",\"velocity_l2_error\":" << std::sqrt(after.error2/after.volume)
          << ",\"velocity_relative_l2_error\":" << std::sqrt(after.error2/std::max(after.reference2, 1.e-300))
          << ",\"velocity_linf_error\":" << after.error_inf
          << ",\"interface_velocity_linf_error\":" << after.interface_error_inf
          << ",\"divergence_before_linf\":" << before.div_inf
          << ",\"divergence_after_linf\":" << after.div_inf
          << ",\"divergence_after_l2\":" << std::sqrt(after.div2/after.volume)
          << ",\"coarse_fine_flux_mismatch\":" << mismatch
          << ",\"projection_idempotence_l2\":" << std::sqrt(repeated.error2/after.volume)
          << ",\"velocity_linf\":" << after.velocity_inf
          << ",\"elapsed_seconds\":" << amrex::second()-start << ",\"peak_rss_mib\":" << peak_mib
          << ",\"passed\":" << (passed ? "true" : "false") << "}\n";
    }
    return passed ? 0 : 2;
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    int result = pilot();
    amrex::Finalize();
    return result;
}
