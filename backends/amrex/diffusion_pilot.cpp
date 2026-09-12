// Forced vector-diffusion verification. No advection or incompressibility coupling yet.
#include <AMReX.H>
#include <AMReX_BoxIterator.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <cmath>
#include <iomanip>
#include <numbers>
#include <string>
#include <sys/resource.h>

using namespace amrex;
constexpr Real pi = std::numbers::pi_v<Real>;
constexpr int components = 3;

struct Norms {
    Real error2=0, energy=0, maximum=0, volume=0;
    Array<Real,components> integral{};
};

int run_diffusion()
{
    ParmParse pp;
    int n=16, levels=3, steps=32, box_size=32;
    Real end_time=.2, viscosity=.01;
    std::string mode="spatial";
    pp.query("n_cell",n); pp.query("levels",levels); pp.query("steps",steps);
    pp.query("max_grid_size",box_size); pp.query("test_case",mode);
    pp.query("t_end",end_time); pp.query("viscosity",viscosity);
    AMREX_ALWAYS_ASSERT(n>=16 && n<=128 && n%8==0 && levels>=1 && levels<=12);
    AMREX_ALWAYS_ASSERT(steps>=1 && steps<=4096 && box_size>=8 && box_size%8==0);
    AMREX_ALWAYS_ASSERT(std::isfinite(end_time) && end_time>0 && end_time<=1);
    AMREX_ALWAYS_ASSERT(std::isfinite(viscosity) && viscosity>0);
    AMREX_ALWAYS_ASSERT(mode=="rest" || mode=="constant" || mode=="decay" || mode=="spatial" || mode=="temporal");
    Real started=amrex::second(), dt=end_time/steps, omega=10;
    Vector<Geometry> geom;
    Vector<BoxArray> grids;
    Vector<DistributionMapping> mapping;
    Vector<MultiFab> state(levels), basis(levels), lap_basis(levels), rhs(levels), applied(levels);
    Vector<iMultiFab> mask(levels);
    RealBox physical({-1.,-1.,-1.},{1.,1.,1.}); int periodic[3]={1,1,1};
    for (int lev=0;lev<levels;++lev) {
        int full_n=n*(1<<lev), lo=(full_n-n)/2;
        Box domain(IntVect(0),IntVect(full_n-1));
        geom.emplace_back(domain,&physical,0,periodic);
        BoxArray boxes(lev==0 ? domain : Box(IntVect(lo),IntVect(lo+n-1)));
        boxes.maxSize(box_size); grids.push_back(boxes); mapping.emplace_back(boxes);
        state[lev].define(boxes,mapping[lev],components,1);
        basis[lev].define(boxes,mapping[lev],components,1);
        lap_basis[lev].define(boxes,mapping[lev],components,0);
        rhs[lev].define(boxes,mapping[lev],components,0);
        applied[lev].define(boxes,mapping[lev],components,0);
        mask[lev].define(boxes,mapping[lev],1,0); mask[lev].setVal(1);
        state[lev].setVal(0); basis[lev].setVal(0);
        Real dx=geom[lev].CellSize(0);
        for (MFIter mfi(state[lev]);mfi.isValid();++mfi) {
            auto u=state[lev].array(mfi), p=basis[lev].array(mfi);
            for (BoxIterator it(mfi.validbox());it.ok();++it) for(int c=0;c<components;++c) {
                auto const& i=it(); Real wave=.25;
                for(int d=0;d<3;++d) {
                    Real k=(1+(d+c)%3)*pi, x=-1+(i[d]+.5)*dx;
                    wave*=std::cos(k*x)*std::sin(k*dx/2)/(k*dx/2);
                }
                p(i,c)=.1*(c+1)+wave; // Exact cell average; nonzero mean tests conservation.
                u(i,c)=mode=="constant" ? .1*(c+1) : mode=="decay" ? p(i,c) : 0;
            }
        }
    }
    for(int lev=0;lev+1<levels;++lev)
        mask[lev]=makeFineMask(grids[lev],mapping[lev],grids[lev+1],IntVect(2),1,0);

    auto configure=[&](MLABecLaplacian& op, Real a, Real b) {
        Array<LinOpBCType,3> bc{LinOpBCType::Periodic,LinOpBCType::Periodic,LinOpBCType::Periodic};
        op.setDomainBC(bc,bc); op.setMaxOrder(3);
        for(int lev=0;lev<levels;++lev) op.setLevelBC(lev,nullptr);
        op.setScalars(a,b);
        for(int lev=0;lev<levels;++lev) {op.setACoeffs(lev,1.);op.setBCoeffs(lev,1.);}
    };
    MLABecLaplacian lap(geom,grids,mapping,LPInfo().setMaxCoarseningLevel(0),{},components);
    configure(lap,0,1);
    MLMG lap_apply(lap); lap_apply.setVerbose(0);
    // Independent of dt. This is used ONLY to isolate temporal truncation error.
    lap_apply.apply(GetVecOfPtrs(lap_basis),GetVecOfPtrs(basis));
    MLABecLaplacian implicit(geom,grids,mapping,LPInfo(),{},components);
    configure(implicit,1,viscosity*dt/2);
    MLMG solver(implicit); solver.setVerbose(0); solver.setBottomVerbose(0); solver.setMaxIter(200);

    auto amplitude=[&](Real t) {return mode=="temporal" ? (1-std::cos(omega*t))/2 : t;};
    auto derivative=[&](Real t) {return mode=="temporal" ? omega*std::sin(omega*t)/2 : 1.;};
    auto measure=[&](Real t) {
        Norms s;
        for(int lev=0;lev<levels;++lev) {
            Real dx=geom[lev].CellSize(0),dv=dx*dx*dx;
            for(MFIter mfi(state[lev]);mfi.isValid();++mfi) {
                auto u=state[lev].const_array(mfi),p=basis[lev].const_array(mfi);
                auto valid=mask[lev].const_array(mfi);
                for(BoxIterator it(mfi.validbox());it.ok();++it) {
                    auto const& i=it(); if(!valid(i)) continue; s.volume+=dv;
                    for(int c=0;c<components;++c) {
                        Real offset=.1*(c+1), exact=0;
                        if(mode=="constant") exact=offset;
                        else if(mode=="decay") exact=offset+(p(i,c)-offset)*std::exp(-viscosity*14*pi*pi*t);
                        else if(mode!="rest") exact=amplitude(t)*p(i,c);
                        Real error=u(i,c)-exact;
                        s.error2+=error*error*dv; s.energy+=u(i,c)*u(i,c)*dv/2;
                        s.maximum=std::max(s.maximum,std::abs(u(i,c)));
                        s.integral[c]+=u(i,c)*dv;
                    }
                }
            }
        }
        ParallelDescriptor::ReduceRealSum(s.error2); ParallelDescriptor::ReduceRealSum(s.energy);
        ParallelDescriptor::ReduceRealSum(s.volume); ParallelDescriptor::ReduceRealMax(s.maximum);
        for(auto& v:s.integral) ParallelDescriptor::ReduceRealSum(v);
        return s;
    };
    Norms initial=measure(0), result=initial;
    Array<Real,components> expected_integral=initial.integral;
    Real mass_defect=0, max_residual=0, max_energy_increase=0;
    for(int step=0;step<steps;++step) {
        Real t=step*dt, a=.5*(amplitude(t)+amplitude(t+dt)), ad=.5*(derivative(t)+derivative(t+dt));
        solver.apply(GetVecOfPtrs(applied),GetVecOfPtrs(state));
        Array<Real,components> impulse{};
        for(int lev=0;lev<levels;++lev) {
            Real dx=geom[lev].CellSize(0),dv=dx*dx*dx;
            for(MFIter mfi(rhs[lev]);mfi.isValid();++mfi) {
                auto b=rhs[lev].array(mfi);
                auto u=state[lev].const_array(mfi),Au=applied[lev].const_array(mfi);
                auto p=basis[lev].const_array(mfi),Lp=lap_basis[lev].const_array(mfi);
                auto valid=mask[lev].const_array(mfi);
                for(BoxIterator it(mfi.validbox());it.ok();++it) for(int c=0;c<components;++c) {
                    auto const& i=it(); Real force=0;
                    if(mode=="spatial") force=ad*p(i,c)+viscosity*a*14*pi*pi*(p(i,c)-.1*(c+1));
                    if(mode=="temporal") force=ad*p(i,c)+viscosity*a*Lp(i,c);
                    b(i,c)=2*u(i,c)-Au(i,c)+dt*force;
                    if(valid(i)) impulse[c]+=dt*force*dv;
                }
            }
        }
        for(int c=0;c<components;++c) {ParallelDescriptor::ReduceRealSum(impulse[c]);expected_integral[c]+=impulse[c];}
        max_residual=std::max(max_residual,solver.solve(GetVecOfPtrs(state),GetVecOfConstPtrs(rhs),1.e-12,1.e-13));
        for(int lev=levels-1;lev>0;--lev) average_down(state[lev],state[lev-1],0,components,IntVect(2));
        Norms next=measure((step+1)*dt);
        max_energy_increase=std::max(max_energy_increase,next.energy-result.energy);
        result=next;
        for(int c=0;c<components;++c) mass_defect=std::max(mass_defect,std::abs(result.integral[c]-expected_integral[c]));
    }
    Real error=std::sqrt(result.error2/result.volume);
    bool passed=std::isfinite(error) && std::isfinite(result.energy) && mass_defect<1.e-9
        && std::abs(result.volume-8)<1.e-10 && max_residual<1.e-9
        && (mode!="rest" || result.maximum==0)
        && (mode!="constant" || error<1.e-10)
        && (mode!="decay" || max_energy_increase<1.e-10);
    struct rusage usage{}; getrusage(RUSAGE_SELF,&usage);
#ifdef __APPLE__
    Real memory=usage.ru_maxrss/1048576.;
#else
    Real memory=usage.ru_maxrss/1024.;
#endif
    amrex::Print()<<std::setprecision(17)<<"NS_DIFFUSION_RESULT {\"schema_version\":1,"
      <<"\"scope\":\"forced vector diffusion only; not Navier-Stokes or the project force\","
      <<"\"source_sha256\":\""<<NS_DIFFUSION_SOURCE_SHA256<<"\",\"amrex_commit\":\""<<NS_AMREX_COMMIT<<"\","
      <<"\"test_case\":\""<<mode<<"\",\"base_n\":"<<n<<",\"levels\":"<<levels<<",\"steps\":"<<steps
      <<",\"max_grid_size\":"<<box_size<<",\"dt\":"<<dt<<",\"t_end\":"<<end_time<<",\"viscosity\":"<<viscosity
      <<",\"components\":3,\"initial_linf\":"<<initial.maximum<<",\"final_linf\":"<<result.maximum
      <<",\"l2_error\":"<<error<<",\"mass_balance_defect\":"<<mass_defect<<",\"max_linear_residual\":"<<max_residual
      <<",\"max_energy_increase\":"<<max_energy_increase<<",\"initial_energy\":"<<initial.energy<<",\"final_energy\":"<<result.energy
      <<",\"composite_volume\":"<<result.volume<<",\"elapsed_seconds\":"<<amrex::second()-started
      <<",\"peak_rss_mib\":"<<memory<<",\"passed\":"<<(passed ? "true" : "false")<<"}\n";
    return passed ? 0 : 2;
}

int main(int argc,char* argv[]) {amrex::Initialize(argc,argv);int status=run_diffusion();amrex::Finalize();return status;}
