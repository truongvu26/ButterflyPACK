// Compare native GPU QR/SVD/WY recompression against independent host BLAS
// products. Include mixed batches, QR/SVD thresholds, zero tau, and scaling.
#include "hodlr_gpu/hodlr_construct.hpp"
#include <mpi.h>
#include <cmath>
#include <complex>
#include <cstdio>
#include <numeric>
#include <random>
#include <fstream>

extern "C" {
void dgemm_(const char*,const char*,const int*,const int*,const int*,const double*,
            const double*,const int*,const double*,const int*,const double*,double*,const int*);
void zgemm_(const char*,const char*,const int*,const int*,const int*,const fmm::gpu::dcomplex*,
            const fmm::gpu::dcomplex*,const int*,const fmm::gpu::dcomplex*,const int*,
            const fmm::gpu::dcomplex*,fmm::gpu::dcomplex*,const int*);
}

namespace bpack { namespace gpu {
struct HodlrConstructTestAccess {
    template<typename T> static void inject(HodlrConstruct<T>& c,int m,int n,int k,
                                            const std::vector<T>& u,const std::vector<T>& v) {
        c.blocks_.emplace_back(); auto& b=c.blocks_.back(); b.m=m; b.n=n;
        c.grow(b,k); b.rank=k;
        fmm::gpu::check_cuda(cudaMemcpy(b.u,u.data(),u.size()*sizeof(T),cudaMemcpyHostToDevice),"test U");
        fmm::gpu::check_cuda(cudaMemcpy(b.vt,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice),"test V");
    }
};
}}
template<typename T> std::vector<T> product(int m,int n,int k,const std::vector<T>& u,const std::vector<T>& v) {
    const T one(1.),zero(0.); const char N='N',trans='T'; std::vector<T> a(size_t(m)*n);
    if constexpr(std::is_same_v<T,double>) dgemm_(&N,&trans,&m,&n,&k,&one,u.data(),&m,v.data(),&n,&zero,a.data(),&m);
    else zgemm_(&N,&trans,&m,&n,&k,&one,u.data(),&m,v.data(),&n,&zero,a.data(),&m);
    return a;
}
template<typename T> struct Input { int m,n,k,kind; std::vector<T> u,v; };
template<typename T> void check_svd(const Input<T>& x,int rank) {
    auto expected=product(x.m,x.n,x.k,x.u,x.v);
    using namespace bpack::gpu;
    auto& heap=DeviceHeap::instance(); const int k=std::min(x.m,x.n);
    T* a=heap.alloc<T>(expected.size()*sizeof(T));
    T* u=heap.alloc<T>(size_t(x.m)*k*sizeof(T));T* v=heap.alloc<T>(size_t(x.n)*k*sizeof(T));
    double* s=heap.alloc<double>(size_t(k)*sizeof(double));
    fmm::gpu::check_cuda(cudaMemcpy(a,expected.data(),expected.size()*sizeof(T),cudaMemcpyHostToDevice),"direct SVD input");
    DeviceSvd<T> svd;double perturbation=svd.svd_device(x.m,x.n,a,s,u,v);
    std::vector<T> hu(size_t(x.m)*k),hv(size_t(x.n)*k);std::vector<double> hs(k);
    fmm::gpu::check_cuda(cudaMemcpy(hu.data(),u,hu.size()*sizeof(T),cudaMemcpyDeviceToHost),"direct SVD U");
    fmm::gpu::check_cuda(cudaMemcpy(hv.data(),v,hv.size()*sizeof(T),cudaMemcpyDeviceToHost),"direct SVD V");
    fmm::gpu::check_cuda(cudaMemcpy(hs.data(),s,hs.size()*sizeof(double),cudaMemcpyDeviceToHost),"direct SVD singular values");
    for(int j=0;j<k;++j) {
        for(int i=0;i<x.m;++i)hu[i+size_t(j)*x.m]=hu[i+size_t(j)*x.m]*T(hs[j]);
        if constexpr(!std::is_same_v<T,double>)for(int i=0;i<x.n;++i)hv[i+size_t(j)*x.n].im*=-1;
    }
    auto actual=product(x.m,x.n,k,hu,hv);long double error=0,norm=0;
    for(size_t i=0;i<actual.size();++i){error+=fmm::gpu::abs2(actual[i]-expected[i]);norm+=fmm::gpu::abs2(expected[i]);}
    double relative=std::sqrt(error/norm);
    std::printf("direct_svd rank=%d m=%d n=%d k=%d err_sigma=%.12g norm=%.12g relative_error=%.12g\n",
                rank,x.m,x.n,k,perturbation,double(std::sqrt(norm)),relative);std::fflush(stdout);
    for(void* p:{static_cast<void*>(s),static_cast<void*>(v),static_cast<void*>(u),static_cast<void*>(a)})heap.free(p);
    if(!std::isfinite(relative)||relative>1.e-7)throw std::runtime_error("Direct GPU SVD failed independent reconstruction");
}
template<typename T> std::vector<Input<T>> inputs() {
    std::mt19937_64 rng(98531); std::uniform_real_distribution<double> rand(-1.,1.);
    auto value=[&]() -> T {
        double re=rand(rng),im=rand(rng);
        if constexpr(std::is_same_v<T,double>) return re;
        else return T(re,im);
    };
    std::vector<Input<T>> out;
    const int ranks[]={32,64,255,256,511,512,640};
    for(int a=0;a<7;++a) {
        int k=ranks[a],kind=a%3,m=kind==1?k:k+13,n=k+31;
        Input<T> x{m,n,k,kind,std::vector<T>(size_t(m)*k),std::vector<T>(size_t(n)*k)};
        for(int j=0;j<k;++j) {
            double scale=kind==2?std::pow(10.,-10.*j/(k-1)):1.;
            for(int i=0;i<m;++i)x.u[i+size_t(j)*m]=value()*scale/std::sqrt(double(m));
            for(int i=0;i<n;++i)x.v[i+size_t(j)*n]=value()/std::sqrt(double(n));
        }
        if(kind==1) for(int j:{0,k/2,k-1})
            std::fill(x.u.begin()+size_t(j)*m,x.u.begin()+size_t(j+1)*m,T(0.));
        out.push_back(std::move(x));
    }
    // Exact-factor fallback uses I*A or A*I. These give zero Householder
    // coefficients with nonzero R diagonals, on both sides of the product.
    for(int k:{200,512})for(int side:{0,1}) {
        const int m=side?k+31:k,n=side?k:k+31;
        Input<T> x{m,n,k,3+side,std::vector<T>(size_t(m)*k),std::vector<T>(size_t(n)*k)};
        for(int j=0;j<k;++j) {
            for(int i=0;i<m;++i)x.u[i+size_t(j)*m]=side?value():T(i==j?1.:0.);
            for(int i=0;i<n;++i)x.v[i+size_t(j)*n]=side?T(i==j?1.:0.):value();
        }
        out.push_back(std::move(x));
    }
    // A square, exactly rank-deficient core is a realistic frontal SVD
    // input. Random full-rank factors do not exercise its null space.
    {
        const int k=576;
        Input<T> x{k,k,k,6,std::vector<T>(size_t(k)*k),std::vector<T>(size_t(k)*k)};
        for(int j=0;j<k;++j) {
            x.u[j+size_t(j)*k]=T(1.);
            if(j<128) {
                if constexpr(std::is_same_v<T,double>)x.v[j+size_t(j)*k]=.1/(1.+.01*j);
                else x.v[j+size_t(j)*k]=T(.1/(1.+.01*j),.01);
            }
        }
        out.push_back(std::move(x));
    }
    return out;
}
template<typename T> void run(const std::vector<Input<T>>& cases,bool batch,int rank,double tol,bool adopt=false) {
    using namespace bpack::gpu;
    for(size_t first=0;first<cases.size();first+=batch?cases.size():1) {
        const int count=batch?int(cases.size()):1;
        HodlrConstruct<T> c; DeviceSvd<T> svd;
        std::vector<int> blocks(count),ranks(count); std::iota(blocks.begin(),blocks.end(),0);
        for(int a=0;a<count;++a) {
            const auto& x=cases[first+a]; HodlrConstructTestAccess::inject(c,x.m,x.n,x.k,x.u,x.v);
        }
        c.recompress(count,blocks.data(),tol,1.e-300,svd,ranks.data());
        if(adopt) {
            c.adopt_output(count,blocks.data(),ranks.data());
            c.recompress(count,blocks.data(),tol,1.e-300,svd,ranks.data());
        }
        std::vector<std::vector<T>> u(count),v(count); std::vector<T*> up(count),vp(count);
        for(int a=0;a<count;++a) {
            const auto& x=cases[first+a]; u[a].resize(size_t(x.m)*ranks[a]);v[a].resize(size_t(x.n)*ranks[a]);
            up[a]=u[a].data();vp[a]=v[a].data();
        }
        c.download(count,blocks.data(),ranks.data(),up.data(),vp.data());
        for(int a=0;a<count;++a) {
            const auto& x=cases[first+a]; auto expected=product(x.m,x.n,x.k,x.u,x.v);
            auto actual=product(x.m,x.n,ranks[a],u[a],v[a]); long double diff=0,norm=0;
            for(size_t i=0;i<actual.size();++i){diff+=fmm::gpu::abs2(actual[i]-expected[i]);norm+=fmm::gpu::abs2(expected[i]);}
            const double error=std::sqrt(diff/norm);
            const double bound=std::max(2.e-8,(adopt?4:2)*std::sqrt(double(x.k))*tol);
            std::printf("recompression rank=%d scalar=%s batch=%d adopt=%d m=%d n=%d k=%d kind=%d retained=%d tol=%.2g error=%.12g bound=%.3g\n",
                rank,std::is_same_v<T,double>?"d":"z",batch,adopt,x.m,x.n,x.k,x.kind,ranks[a],tol,error,bound);std::fflush(stdout);
            if(!std::isfinite(error)||error>bound)throw std::runtime_error("GPU recompression changed the represented matrix");
        }
    }
}
int main(int argc,char** argv) {
    MPI_Init(&argc,&argv); int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    try {
        fmm::gpu::share_device(MPI_COMM_WORLD);
        auto d=inputs<double>();auto z=inputs<fmm::gpu::dcomplex>();
        const bool direct=argc>1 && std::string(argv[1])=="--svd";
        if(argc>1) {
            d.clear();z.clear();
            if(direct && argc<3)throw std::runtime_error("--svd needs a factor capture");
            for(int f=direct?2:1;f<argc;++f) {
                std::ifstream file(argv[f],std::ios::binary);uint64_t h[5];file.read(reinterpret_cast<char*>(h),sizeof(h));
                if(!file || h[0]!=0x524f544341465042ULL || !h[1] || !h[2] || !h[3] ||
                   h[1]>100000 || h[2]>100000 || h[3]>std::min(h[1],h[2]) ||
                   (h[4]!=8 && h[4]!=16) || (h[1]+h[2])*h[3]*h[4]>(uint64_t{64}<<20))
                    throw std::runtime_error("Invalid GPU factor capture");
                auto read=[&](auto& cases) {
                    using T=typename std::decay_t<decltype(cases)>::value_type;
                    T x{int(h[1]),int(h[2]),int(h[3]),5,{},{}};
                    x.u.resize(h[1]*h[3]);x.v.resize(h[2]*h[3]);
                    file.read(reinterpret_cast<char*>(x.u.data()),x.u.size()*h[4]);
                    file.read(reinterpret_cast<char*>(x.v.data()),x.v.size()*h[4]);
                    if(!file)throw std::runtime_error("Truncated GPU factor capture");
                    cases.push_back(std::move(x));
                };
                if(h[4]==8)read(d);else read(z);
            }
        }
        if(direct) {for(auto& x:d)check_svd(x,rank);for(auto& x:z)check_svd(x,rank);}
        else {
            if(argc==1) {
                for(auto& x:d)if(x.kind==6 || (x.k==200 && x.kind>=3))check_svd(x,rank);
                for(auto& x:z)if(x.kind==6 || (x.k==200 && x.kind>=3))check_svd(x,rank);
            }
            for(double tol:{1.e-5,1.e-10}) {
                if(!d.empty()) {run(d,true,rank,tol);run(d,false,rank,tol);run(d,true,rank,tol,true);}
                if(!z.empty()) {run(z,true,rank,tol);run(z,false,rank,tol);run(z,true,rank,tol,true);}
            }
        }
        std::printf("GPU recompression regression rank=%d PASS\n",rank);
    } catch(const std::exception& error) {
        std::fprintf(stderr,"rank=%d %s\n",rank,error.what());MPI_Abort(MPI_COMM_WORLD,1);
    }
    MPI_Finalize();
}
