#ifndef NS_CASE_H
#define NS_CASE_H

// CPU, constant-density validation adapter for the coupled incflo integrator.
// These manufactured tests are NOT the paper-surrogate force.
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_TagBox.H>
#include <cmath>
#include <iomanip>
#include <sys/resource.h>
#ifndef NS_ADAPTER_SHA256
#define NS_ADAPTER_SHA256 "not-built-as-integrator"
#endif

namespace ns_case {
using namespace amrex;
constexpr Real pi = 3.1415926535897932384626433832795;
struct Options {
    std::string force = "none";
    Real max_dt = 1e100;
    Vector<Real> widths;
    bool diagnostics = false;
    Options() {
        ParmParse pp("ns");
        pp.query("force", force);
        pp.query("max_dt", max_dt);
        pp.queryarr("refine_half_width", widths);
        pp.query("diagnostics", diagnostics);
        if (force != "none" && force != "rest" && force != "oscillation" && force != "shear")
            Abort("ns.force must be none, rest, oscillation, or shear; paper forcing is not implemented");
        if (!std::isfinite(max_dt) || max_dt <= 0) Abort("ns.max_dt must be finite and positive");
        for (int l=0; l<widths.size(); ++l)
            if (!std::isfinite(widths[l]) || widths[l] <= 0 || widths[l] >= (l ? widths[l-1] : 1.0))
                Abort("fixed refinement widths must decrease inside [-1,1]^3");
    }
};
inline Options const& options() { static Options o; return o; }
inline void validate(Real fixed_dt, int regrid_int, bool constant_density, Real density) {
    auto const& o=options();
    if (fixed_dt > 0 && (o.force != "none" || o.max_dt < 1e99))
        Abort("use ns.max_dt, not the upstream fixed_dt CFL override");
    if (!o.widths.empty() && regrid_int > 0)
        Abort("pre-refined tests require amr.regrid_int <= 0");
    if (o.force != "none" && (!constant_density || density != 1.0))
        Abort("the manufactured acceleration adapter requires constant unit density");
}
inline Real limit_dt(Real /*time*/, Real dt) { return std::min(dt, options().max_dt); }

inline Real force_value(std::string const& mode, Real x, Real y, Real z, Real t, int c, Real nu) {
    if (mode == "oscillation") return 0.1*(c+1)*5.0*std::sin(10*t);
    if (mode != "shear") return 0;
    Real coord[3]={y,z,x};
    Real advector[3]={z,x,y};
    return (1+nu*pi*pi*t)*std::sin(pi*coord[c])
        + t*t*pi*std::sin(pi*advector[c])*std::cos(pi*coord[c]);
}
inline Real exact_value(int problem, Real x, Real y, Real z, Real t, int c, Real nu) {
    auto const& mode=options().force;
    if (mode == "rest") return 0;
    if (mode == "oscillation") return 0.1*(c+1)*0.5*(1-std::cos(10*t));
    if (mode == "shear") { Real coord[3]={y,z,x}; return t*std::sin(pi*coord[c]); }
    if (problem == 2) {
        if (c==0) return 1-std::cos(pi*(x-t))*std::sin(pi*(y-t))*std::exp(-2*pi*pi*nu*t);
        if (c==1) return 1+std::sin(pi*(x-t))*std::cos(pi*(y-t))*std::exp(-2*pi*pi*nu*t);
        return 0;
    }
    Abort("diagnostic exact reference is unavailable for this case");
    return 0;
}
inline void add_force(MultiFab& f, Geometry const& g, Real time, Real end, Real nu) {
    auto const& mode=options().force;
    if (mode == "none" || mode == "rest") return;
    auto dx=g.CellSizeArray(); auto lo=g.ProbLoArray();
    for (MFIter mfi(f); mfi.isValid(); ++mfi) {
        auto a=f.array(mfi);
        LoopOnCpu(mfi.validbox(), [&](int i,int j,int k) {
            Real x=lo[0]+(i+0.5)*dx[0], y=lo[1]+(j+0.5)*dx[1], z=lo[2]+(k+0.5)*dx[2];
            for (int c=0;c<3;++c) {
                Real value=force_value(mode,x,y,z,time,c,nu);
                if (end >= 0) value=0.5*(value+force_value(mode,x,y,z,end,c,nu));
                a(i,j,k,c)+=value;
            }
        });
    }
}
inline void tag_fixed_region(int level, TagBoxArray& tags, Geometry const& g) {
    auto const& widths=options().widths;
    if (level >= widths.size()) return;
    auto dx=g.CellSizeArray(); auto lo=g.ProbLoArray();
    Real width=widths[level];
    for (int c=0;c<3;++c)
        if (g.ProbLo(c)!=-1 || g.ProbHi(c)!=1 || !g.isPeriodic(c))
            Abort("fixed refinement requires full periodic [-1,1]^3");
    for (MFIter mfi(tags); mfi.isValid(); ++mfi) {
        auto a=tags.array(mfi);
        LoopOnCpu(mfi.validbox(), [&](int i,int j,int k) {
            if (std::abs(lo[0]+(i+0.5)*dx[0]) < width &&
                std::abs(lo[1]+(j+0.5)*dx[1]) < width &&
                std::abs(lo[2]+(k+0.5)*dx[2]) < width) a(i,j,k)=TagBox::SET;
        });
    }
}
inline void diagnose(Vector<MultiFab const*> const& vel, Vector<Geometry> const& geom,
                     Vector<IntVect> const& ratio, int problem, Real time, Real dt, int step, Real nu) {
    if (!options().diagnostics) return;
    Real volume=0, sumsq=0, maxerr=0, peak=0, energy=0;
    Real integral[3]={0,0,0};
    Long stored=0,active=0;
    for (int lev=0;lev<vel.size();++lev) {
        auto const& v=*vel[lev];
        iMultiFab mask;
        if (lev+1 < vel.size()) mask=makeFineMask(v.boxArray(),v.DistributionMap(),vel[lev+1]->boxArray(),ratio[lev],1,0);
        else { mask.define(v.boxArray(),v.DistributionMap(),1,0); mask.setVal(1); }
        auto dx=geom[lev].CellSizeArray(); auto lo=geom[lev].ProbLoArray();
        Real dv=dx[0]*dx[1]*dx[2];
        stored+=v.boxArray().numPts();
        Long active_before=active;
        for (MFIter mfi(v);mfi.isValid();++mfi) {
            auto a=v.const_array(mfi); auto m=mask.const_array(mfi);
            LoopOnCpu(mfi.validbox(),[&](int i,int j,int k) {
                if (!m(i,j,k)) return;
                ++active; Real speed2=0;
                Real x=lo[0]+(i+0.5)*dx[0], y=lo[1]+(j+0.5)*dx[1], z=lo[2]+(k+0.5)*dx[2];
                for (int c=0;c<3;++c) {
                    Real u=a(i,j,k,c);
                    if (!std::isfinite(u)) Abort("non-finite coupled solution");
                    Real e=u-exact_value(problem,x,y,z,time,c,nu);
                    sumsq+=e*e*dv; maxerr=std::max(maxerr,std::abs(e));
                    speed2+=u*u; integral[c]+=u*dv;
                }
                peak=std::max(peak,std::sqrt(speed2)); energy+=0.5*speed2*dv;
            });
        }
        // Integer counts avoid O(cell count) roundoff for non-binary spacings.
        volume+=Real(active-active_before)*dv;
    }
    Real domain_volume=1;
    for (int c=0;c<3;++c) domain_volume*=geom[0].ProbHi(c)-geom[0].ProbLo(c);
    if (std::abs(volume-domain_volume)>1e-10) Abort("composite diagnostic volume does not cover the domain");
    // Current target is explicitly serial CPU. No misleading distributed reduction.
    if (ParallelDescriptor::NProcs()!=1) Abort("diagnostic reductions require the serial validation build");
    struct rusage usage{}; getrusage(RUSAGE_SELF,&usage);
    Real rss=usage.ru_maxrss;
#ifdef __APPLE__
    rss/=1024*1024;
#else
    rss/=1024;
#endif
    Print()<<std::setprecision(17)<<"NS_INCFLO_RESULT {\"schema_version\":1,\"adapter_sha256\":\""
        <<NS_ADAPTER_SHA256<<"\",\"case\":\""
        <<options().force<<"\",\"problem\":"<<problem<<",\"step\":"<<step<<",\"time\":"<<time
        <<",\"dt\":"<<dt<<",\"levels\":"<<vel.size()<<",\"stored_cells\":"<<stored
        <<",\"active_cells\":"<<active<<",\"volume\":"<<volume<<",\"l2_error\":"<<std::sqrt(sumsq/volume)
        <<",\"linf_error\":"<<maxerr<<",\"peak_speed\":"<<peak<<",\"energy\":"<<energy
        <<",\"velocity_integral\":["<<integral[0]<<","<<integral[1]<<","<<integral[2]
        <<"],\"peak_rss_mib\":"<<rss<<"}\n";
}
}
#endif
