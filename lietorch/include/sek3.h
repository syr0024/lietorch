
#ifndef SEK3_HEADER
#define SEK3_HEADER

#include <stdio.h>
#include <Eigen/Dense>
#include <Eigen/Geometry> 

#include "common.h"
#include "so3.h"


template <typename Scalar>
class SEK3 {
  public:
    const static int constexpr k = 6;
    const static int constexpr K = 3+3*k; // manifold dimension
    const static int constexpr N = 4+3*k; // embedding dimension
    const static int constexpr M = 3+k; // matrix dimension

    using Vector3 = Eigen::Matrix<Scalar,3,1>;
    using Vector4 = Eigen::Matrix<Scalar,4,1>;
    using Vector6 = Eigen::Matrix<Scalar,6,1>;
    using Vector3k = Eigen::Matrix<Scalar,3*k,1>;
    using Matrix3 = Eigen::Matrix<Scalar,3,3>;
    using Matrix4 = Eigen::Matrix<Scalar,4,4>;
    using MatrixM = Eigen::Matrix<Scalar,M,M>;

    using Tangent = Eigen::Matrix<Scalar,K,1>;
    using Point = Eigen::Matrix<Scalar,3,1>;
    using Point4 = Eigen::Matrix<Scalar,4,1>;
    using PointM = Eigen::Matrix<Scalar,M,1>;
    using Data = Eigen::Matrix<Scalar,N,1>;
    using Transformation = Eigen::Matrix<Scalar,M,M>;
    using Adjoint = Eigen::Matrix<Scalar,K,K>;

    EIGEN_DEVICE_FUNC SEK3() { translation = Vector3k::Zero(); }

    EIGEN_DEVICE_FUNC SEK3(SO3<Scalar> const& so3, Vector3k const& t) : so3(so3), translation(t) {};

    EIGEN_DEVICE_FUNC SEK3(const Scalar *data) : so3(data), translation(data+4) {};

    EIGEN_DEVICE_FUNC SEK3<Scalar> inv() {
      Vector3k trans_inv;
      for (int i=0; i<k; i++) {
        trans_inv.template segment<3>(3*i) = -(so3.inv()*translation.template segment<3>(3*i));
      }
      return SEK3(so3.inv(), trans_inv);
    }

    EIGEN_DEVICE_FUNC Data data() const {
      Data data_vec; data_vec << so3.data(), translation;
      return data_vec;
    }

    EIGEN_DEVICE_FUNC SEK3<Scalar> operator*(SEK3<Scalar> const& other) {
      Vector3k trans_op;
      for (int i=0; i<k; i++) {
        trans_op.template segment<3>(3*i) = translation.template segment<3>(3*i) + so3 * other.translation.template segment<3>(3*i);
      }
      return SEK3(so3 * other.so3, trans_op);
    }

    EIGEN_DEVICE_FUNC Point operator*(Point const& p) const {
      return so3 * p + translation.template segment<3>(3);
    }

    EIGEN_DEVICE_FUNC PointM act4(PointM const& p) const {
      PointM p1;
      Point p_up = so3 * p.template segment<3>(0);
      for (int i=0; i<k; i++){
        p_up = p_up + translation.template segment<3>(3*i) * p(3+i);
      }
      p1 << p_up, p.template segment<k>(3);
      return p1;
    }

    EIGEN_DEVICE_FUNC Adjoint Adj() const {
      Matrix3 R = so3.Matrix();
      Matrix3 tx = Matrix3::Zero();
      Matrix3 Zer = Matrix3::Zero();
    
      Adjoint Ad = Adjoint::Zero(); Ad.template block<3,3>(0,0) = R;
      for (int i=0; i<k; i++) {
        tx = SO3<Scalar>::hat(translation.template segment<3>(3*i));
        Ad.template block<3,3>((i+1)*3,0) = tx*R;
        Ad.template block<3,3>((i+1)*3,(i+1)*3) = R;
      }
      
      return Ad;
    }

    EIGEN_DEVICE_FUNC Transformation Matrix() const {
      Transformation T = Transformation::Identity();
      T.template block<3,3>(0,0) = so3.Matrix();
      for (int i=0; i<k; i++) {
        T.template block<3,1>(0,3*(i+1)) = translation.template segment<3>(3*i);
      }
      return T;
    }

    EIGEN_DEVICE_FUNC MatrixM Matrix4x4() const {
      return Matrix();
    }

    EIGEN_DEVICE_FUNC Tangent Adj(Tangent const& a) const {
      return Adj() * a;
    }

    EIGEN_DEVICE_FUNC Tangent AdjT(Tangent const& a) const {
      return Adj().transpose() * a;
    }
    
    EIGEN_DEVICE_FUNC static Transformation hat(Tangent const& phi_tau) {
      Vector3 phi = phi_tau.template segment<3>(3);
      Vector3 tau = phi_tau.template segment<3>(0);

      Transformation PhiTau = Transformation::Zero();
      PhiTau.template block<3,3>(0,0) = SO3<Scalar>::hat(phi);
      for (int i=0; i<k; i++){
        PhiTau.template block<3,1>(0,3*(i+1)) = tau;
      }
      
      return PhiTau;
    }

    EIGEN_DEVICE_FUNC static Adjoint adj(Tangent const& phi_tau) {
      Vector3 phi = phi_tau.template segment<3>(0);
      Vector3k tau = phi_tau.template segment<3*k>(3);
    
      Matrix3 Phi = SO3<Scalar>::hat(phi);
      Adjoint ad = Adjoint::Zero(); ad.template block<3,3>(0,0) = Phi;
      for (int i=0; i<k; i++) {
        ad.template block<3,3>((i+1)*3,0) = SO3<Scalar>::hat(tau.template segment<3>(3*i));
        ad.template block<3,3>((i+1)*3,(i+1)*3) = Phi;
      }

      return ad;
    }

    // 일반 tensor matrix를 Liegroup 상에 project 해줄 때 필요한 함수
    EIGEN_DEVICE_FUNC Eigen::Matrix<Scalar,N,N> orthogonal_projector() const {
      // jacobian action on a point
      Eigen::Matrix<Scalar,N,N> J = Eigen::Matrix<Scalar,N,N>::Zero();
      J.template block<4,4>(0,0) = so3.orthogonal_projector();
      for (int i=0; i<k; i++) {
        J.template block<3,3>(4+3*i,3+3*i) = Matrix3::Identity();
        J.template block<3,3>(4+3*i,0) = SO3<Scalar>::hat(-translation.template segment<3>(3*i));
      }

      return J;
    }

    EIGEN_DEVICE_FUNC Tangent Log() const {
      Vector3 phi = so3.Log();      
      Matrix3 Vinv = SO3<Scalar>::left_jacobian_inverse(phi);

      Tangent phi_tau = Tangent::Zero();
      phi_tau.template segment<3>(0) = phi;
      for (int i=0; i<k; i++) {
        phi_tau.template segment<3>(3*(i+1)) = Vinv * translation.template segment<3>(3*i);
      }

      return phi_tau;
    }

    EIGEN_DEVICE_FUNC static SEK3<Scalar> Exp(Tangent const& phi_tau) {
      Vector3 phi = phi_tau.template segment<3>(0);
      Vector3k tau = phi_tau.template segment<3*k>(3);

      SO3<Scalar> so3 = SO3<Scalar>::Exp(phi);
      Vector3k t = Vector3k::Zero();
      for (int i=0; i<k; i++) {
        t.template segment<3>(3*i) = SO3<Scalar>::left_jacobian(phi) * tau.template segment<3>(3*i);
      }

      return SEK3<Scalar>(so3, t);
    }

    EIGEN_DEVICE_FUNC static Matrix3 calcQ(Vector6 const& tau_phi) {
      // Q matrix
      Vector3 tau = tau_phi.template segment<3>(0);
      Vector3 phi = tau_phi.template segment<3>(3);
      Matrix3 Tau = SO3<Scalar>::hat(tau);
      Matrix3 Phi = SO3<Scalar>::hat(phi);

      Scalar theta = phi.norm();
      Scalar theta_pow2 = theta * theta;
      Scalar theta_pow4 = theta_pow2 * theta_pow2;

      Scalar coef1 = (theta < EPS) ?
        Scalar(1.0/6.0) - Scalar(1.0/120.0) * theta_pow2 : 
        (theta - sin(theta)) / (theta_pow2 * theta);

      Scalar coef2 = (theta < EPS) ?
        Scalar(1.0/24.0) - Scalar(1.0/720.0) * theta_pow2 : 
        (theta_pow2 + 2*cos(theta) - 2) / (2 * theta_pow4);

      Scalar coef3 = (theta < EPS) ?
        Scalar(1.0/120.0) - Scalar(1.0/2520.0) * theta_pow2 : 
        (2*theta - 3*sin(theta) + theta*cos(theta)) / (2 * theta_pow4 * theta);

      Matrix3 Q = Scalar(0.5) * Tau + 
        coef1 * (Phi*Tau + Tau*Phi + Phi*Tau*Phi) +
        coef2 * (Phi*Phi*Tau + Tau*Phi*Phi - 3*Phi*Tau*Phi) + 
        coef3 * (Phi*Tau*Phi*Phi + Phi*Phi*Tau*Phi);

      return Q;
    }
    
    EIGEN_DEVICE_FUNC static Adjoint left_jacobian(Tangent const& phi_tau) {
      // left jacobian
      Vector3 phi = phi_tau.template segment<3>(0);
      Matrix3 J = SO3<Scalar>::left_jacobian(phi);
      Adjoint JKxK = Adjoint::Zero();
      JKxK.template block<3,3>(0,0) = J;
      for (int i=0; i<k; i++) {
        JKxK.template block<3,3>(3*(i+1),3*(i+1)) = J;
        Vector6 tau_phi = Vector6::Zero();
        tau_phi << phi_tau.template segment<3>(3*(i+1)), phi;
        Matrix3 Q = SEK3<Scalar>::calcQ(tau_phi);
        JKxK.template block<3,3>(3*(i+1),0) = Q;
      }

      return JKxK;
    }

    EIGEN_DEVICE_FUNC static Adjoint left_jacobian_inverse(Tangent const& phi_tau) {
      // left jacobian inverse
      Vector3 phi = phi_tau.template segment<3>(0);
      Matrix3 Jinv = SO3<Scalar>::left_jacobian_inverse(phi);
      Adjoint JKxK = Adjoint::Zero();
      JKxK.template block<3,3>(0,0) = Jinv;
      for (int i=0; i<k; i++) {
        JKxK.template block<3,3>(3*(i+1),3*(i+1)) = Jinv;
        Vector6 tau_phi = Vector6::Zero();
        tau_phi << phi_tau.template segment<3>(3*(i+1)), phi;
        Matrix3 Q = SEK3<Scalar>::calcQ(tau_phi);
        JKxK.template block<3,3>(3*(i+1),0) = -Jinv * Q * Jinv;
      }

      return JKxK;
    }
    
    // TODO: 수정 안함
    EIGEN_DEVICE_FUNC static Eigen::Matrix<Scalar,3,K> act_jacobian(Point const& p) {
      // jacobian action on a point
      Eigen::Matrix<Scalar,3,K> J = Eigen::Matrix<Scalar,3,K>::Zero();
      J.template block<3,3>(0,0) = Matrix3::Identity();
      J.template block<3,3>(0,3) = SO3<Scalar>::hat(-p);
      return J;
    }
    
    EIGEN_DEVICE_FUNC static Eigen::Matrix<Scalar,M,K> act4_jacobian(PointM const& p) {
      // jacobian action on a point
      Eigen::Matrix<Scalar,M,K> J = Eigen::Matrix<Scalar,M,K>::Zero();
      J.template block<3,3>(0,0) = SO3<Scalar>::hat(-p.template segment<3>(0));
      for (int i=0; i<k; i++) {
        J.template block<3,3>(0,3*(i+1)) = p(3+i) * Matrix3::Identity();
      }
      return J;
    }




  private:
    SO3<Scalar> so3;
    Vector3k translation;

};

#endif

