/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: Global_controller_fcn_earth_rf_journal.c
 *
 * MATLAB Coder version            : 5.4
 * C/C++ source code generated on  : 04-Jul-2022 00:11:40
 */

/* Include Files */
#include "Global_controller_fcn_earth_rf_journal.h"
#include "rt_nonfinite.h"
#include "coder_posix_time.h"
#include "omp.h"
#include "rt_nonfinite.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/* Type Definitions */
#ifndef typedef_struct_T
#define typedef_struct_T

typedef struct {
  double grad[13];
  double Hx[12];
  bool hasLinear;
  int nvar;
  int maxVar;
  double beta;
  double rho;
  int objtype;
  int prev_objtype;
  int prev_nvar;
  bool prev_hasLinear;
  double gammaScalar;
} struct_T;

#endif                                 /* typedef_struct_T */

#ifndef typedef_b_struct_T
#define typedef_b_struct_T

typedef struct {
  double penaltyParam;
  double threshold;
  int nPenaltyDecreases;
  double linearizedConstrViol;
  double initFval;
  double initConstrViolationEq;
  double initConstrViolationIneq;
  double phi;
  double phiPrimePlus;
  double phiFullStep;
  double feasRelativeFactor;
  double nlpPrimalFeasError;
  double nlpDualFeasError;
  double nlpComplError;
  double firstOrderOpt;
  bool hasObjective;
} b_struct_T;

#endif                                 /* typedef_b_struct_T */

#ifndef typedef_c_struct_T
#define typedef_c_struct_T

typedef struct {
  bool gradOK;
  bool fevalOK;
  bool done;
  bool stepAccepted;
  bool failedLineSearch;
  int stepType;
} c_struct_T;

#endif                                 /* typedef_c_struct_T */

#ifndef typedef_captured_var
#define typedef_captured_var

typedef struct {
  double contents;
} captured_var;

#endif                                 /* typedef_captured_var */

#ifndef typedef_b_captured_var
#define typedef_b_captured_var

typedef struct {
  double contents[6];
} b_captured_var;

#endif                                 /* typedef_b_captured_var */

#ifndef typedef_d_struct_T
#define typedef_d_struct_T

typedef struct {
  captured_var *W_act_motor;
  captured_var *gamma_quadratic_du;
  captured_var *desired_motor_value;
  captured_var *gain_motor;
  captured_var *W_dv_2;
  captured_var *gamma_quadratic_dv;
  b_captured_var *dv_global;
  captured_var *S;
  captured_var *V;
  captured_var *rho;
  captured_var *Beta;
  captured_var *Theta;
  captured_var *flight_path_angle;
  captured_var *K_Cd;
  captured_var *Cl_alpha;
  captured_var *Cd_zero;
  captured_var *Psi;
  captured_var *Phi;
  captured_var *K_p_T;
  captured_var *gain_el;
  captured_var *gain_az;
  captured_var *m;
  captured_var *W_act_tilt_el;
  captured_var *desired_el_value;
  captured_var *W_act_tilt_az;
  captured_var *desired_az_value;
  captured_var *W_dv_3;
  captured_var *W_dv_1;
  captured_var *W_dv_5;
  captured_var *I_zz;
  captured_var *p;
  captured_var *r;
  captured_var *I_xx;
  captured_var *I_yy;
  captured_var *l_z;
  captured_var *K_p_M;
  captured_var *Cm_zero;
  captured_var *wing_chord;
  captured_var *l_4;
  captured_var *l_3;
  captured_var *Cm_alpha;
  captured_var *W_dv_6;
  captured_var *q;
  captured_var *l_1;
  captured_var *l_2;
  captured_var *W_dv_4;
} d_struct_T;

#endif                                 /* typedef_d_struct_T */

#ifndef typedef_nested_function
#define typedef_nested_function

typedef struct {
  d_struct_T workspace;
} nested_function;

#endif                                 /* typedef_nested_function */

#ifndef typedef_e_struct_T
#define typedef_e_struct_T

typedef struct {
  double workspace_double[325];
  int workspace_int[25];
  int workspace_sort[25];
} e_struct_T;

#endif                                 /* typedef_e_struct_T */

#ifndef typedef_f_struct_T
#define typedef_f_struct_T

typedef struct {
  int ldq;
  double QR[625];
  double Q[625];
  int jpvt[25];
  int mrows;
  int ncols;
  double tau[25];
  int minRowCol;
  bool usedPivoting;
} f_struct_T;

#endif                                 /* typedef_f_struct_T */

#ifndef typedef_g_struct_T
#define typedef_g_struct_T

typedef struct {
  double FMat[625];
  int ldm;
  int ndims;
  int info;
  double scaleFactor;
  bool ConvexCheck;
  double regTol_;
  double workspace_;
  double workspace2_;
} g_struct_T;

#endif                                 /* typedef_g_struct_T */

#ifndef typedef_h_struct_T
#define typedef_h_struct_T

typedef struct {
  char SolverName[7];
  int MaxIterations;
  double StepTolerance;
  double OptimalityTolerance;
  double ConstraintTolerance;
  double ObjectiveLimit;
  double PricingTolerance;
  double ConstrRelTolFactor;
  double ProbRelTolFactor;
  bool RemainFeasible;
  bool IterDisplayQP;
} h_struct_T;

#endif                                 /* typedef_h_struct_T */

#ifndef typedef_i_struct_T
#define typedef_i_struct_T

typedef struct {
  int nVarMax;
  int mNonlinIneq;
  int mNonlinEq;
  int mIneq;
  int mEq;
  int iNonIneq0;
  int iNonEq0;
  double sqpFval;
  double sqpFval_old;
  double xstarsqp[12];
  double xstarsqp_old[12];
  double grad[13];
  double grad_old[13];
  int FunctionEvaluations;
  int sqpIterations;
  int sqpExitFlag;
  double lambdasqp[25];
  double lambdaStopTest[25];
  double lambdaStopTestPrev[25];
  double steplength;
  double delta_x[13];
  double socDirection[13];
  int workingset_old[25];
  double gradLag[13];
  double delta_gradLag[13];
  double xstar[13];
  double fstar;
  double firstorderopt;
  double lambda[25];
  int state;
  double maxConstr;
  int iterations;
  double searchDir[13];
} i_struct_T;

#endif                                 /* typedef_i_struct_T */

#ifndef typedef_j_struct_T
#define typedef_j_struct_T

typedef struct {
  nested_function objfun;
  double f_1;
  double f_2;
  int nVar;
  int mIneq;
  int mEq;
  int numEvals;
  bool SpecifyObjectiveGradient;
  bool SpecifyConstraintGradient;
  bool isEmptyNonlcon;
  bool hasLB[12];
  bool hasUB[12];
  bool hasBounds;
  int FiniteDifferenceType;
} j_struct_T;

#endif                                 /* typedef_j_struct_T */

#ifndef typedef_k_struct_T
#define typedef_k_struct_T

typedef struct {
  int mConstr;
  int mConstrOrig;
  int mConstrMax;
  int nVar;
  int nVarOrig;
  int nVarMax;
  int ldA;
  double lb[13];
  double ub[13];
  int indexLB[13];
  int indexUB[13];
  int indexFixed[13];
  int mEqRemoved;
  double ATwset[325];
  double bwset[25];
  int nActiveConstr;
  double maxConstrWorkspace[25];
  int sizes[5];
  int sizesNormal[5];
  int sizesPhaseOne[5];
  int sizesRegularized[5];
  int sizesRegPhaseOne[5];
  int isActiveIdx[6];
  int isActiveIdxNormal[6];
  int isActiveIdxPhaseOne[6];
  int isActiveIdxRegularized[6];
  int isActiveIdxRegPhaseOne[6];
  bool isActiveConstr[25];
  int Wid[25];
  int Wlocalidx[25];
  int nWConstr[5];
  int probType;
  double SLACK0;
} k_struct_T;

#endif                                 /* typedef_k_struct_T */

#ifndef typedef_l_struct_T
#define typedef_l_struct_T

typedef struct {
  nested_function objfun;
  int nVar;
  int mCineq;
  int mCeq;
  bool NonFiniteSupport;
  bool SpecifyObjectiveGradient;
  bool SpecifyConstraintGradient;
  bool ScaleProblem;
} l_struct_T;

#endif                                 /* typedef_l_struct_T */

/* Variable Definitions */
static double freq;
static bool freq_not_empty;
static coderTimespec savedTime;
static bool savedTime_not_empty;
omp_nest_lock_t Global_controller_fcn_earth_rf_journal_nestLockGlobal;
static bool isInitialized_Global_controller_fcn_earth_rf_journal = false;

/* Function Declarations */
static bool BFGSUpdate(int nvar, double Bk[144], const double sk[13], double yk
  [13], double workspace[325]);
static void PresolveWorkingSet(i_struct_T *solution, e_struct_T *memspace,
  k_struct_T *workingset, f_struct_T *qrmanager);
static void RemoveDependentIneq_(k_struct_T *workingset, f_struct_T *qrmanager,
  e_struct_T *memspace, double tolfactor);
static void addBoundToActiveSetMatrix_(k_struct_T *obj, int TYPE, int idx_local);
static void b_computeGradLag(double workspace[325], int nVar, const double grad
  [13], const int finiteFixed[13], int mFixed, const int finiteLB[13], int mLB,
  const int finiteUB[13], int mUB, const double lambda[25]);
static void b_driver(const double lb[12], const double ub[12], i_struct_T
                     *TrialState, b_struct_T *MeritFunction, const l_struct_T
                     *FcnEvaluator, e_struct_T *memspace, k_struct_T *WorkingSet,
                     double Hessian[144], f_struct_T *QRManager, g_struct_T
                     *CholManager, struct_T *QPObjective);
static double b_maxConstraintViolation(const k_struct_T *obj, const double x[13]);
static void b_test_exit(c_struct_T *Flags, e_struct_T *memspace, b_struct_T
  *MeritFunction, const k_struct_T *WorkingSet, i_struct_T *TrialState,
  f_struct_T *QRManager, const double lb[12], const double ub[12]);
static void b_timeKeeper(double *outTime_tv_sec, double *outTime_tv_nsec);
static void b_xaxpy(int n, double a, const double x[72], int ix0, double y[12],
                    int iy0);
static double b_xdotc(int n, const double x[36], int ix0, const double y[36],
                      int iy0);
static void b_xgemm(int m, int n, int k, const double A[625], int ia0, const
                    double B[325], double C[625]);
static double b_xnrm2(int n, const double x[13]);
static void b_xrot(double x[72], int ix0, int iy0, double c, double s);
static void b_xswap(double x[72], int ix0, int iy0);
static void c_Basic_inversion_controller_fc(double K_p_T, double K_p_M, double m,
  double I_xx, double I_yy, double I_zz, double l_1, double l_2, double l_3,
  double l_4, double l_z, double Phi, double Theta, double Psi, double Omega_1,
  double Omega_2, double Omega_3, double Omega_4, double b_1, double b_2, double
  b_3, double b_4, double g_1, double g_2, double g_3, double g_4, double
  max_omega, double min_omega, double max_b, double min_b, double max_g, double
  min_g, double dv[6], double p, double q, double r, double Cm_zero, double
  Cl_alpha, double Cd_zero, double K_Cd, double Cm_alpha, double rho, double V,
  double S, double wing_chord, double flight_path_angle, double Beta, double
  desired_motor_value, double desired_el_value, double desired_az_value, double
  u_out[12], double residuals[6], double *elapsed_time, double *exitflag);
static void c_WLS_controller_fcn_earth_rf_j(double K_p_T, double K_p_M, double m,
  double I_xx, double I_yy, double I_zz, double l_1, double l_2, double l_3,
  double l_4, double l_z, double Phi, double Theta, double Psi, double Omega_1,
  double Omega_2, double Omega_3, double Omega_4, double b_1, double b_2, double
  b_3, double b_4, double g_1, double g_2, double g_3, double g_4, double
  W_act_motor_const, double W_act_motor_speed, double W_act_tilt_el_const,
  double W_act_tilt_el_speed, double W_act_tilt_az_const, double
  W_act_tilt_az_speed, double W_dv_1, double W_dv_2, double W_dv_3, double
  W_dv_4, double W_dv_5, double W_dv_6, double max_omega, double min_omega,
  double max_b, double min_b, double max_g, double min_g, double dv[6], double p,
  double q, double r, double Cm_zero, double Cl_alpha, double Cd_zero, double
  K_Cd, double Cm_alpha, double rho, double V, double S, double wing_chord,
  double flight_path_angle, double max_alpha, double Beta, double gammasq,
  double desired_motor_value, double desired_el_value, double desired_az_value,
  double u_out[12], double residuals[6], double *elapsed_time, double
  *N_iterations, double *N_evaluation, double *exitflag);
static void c_compute_acc_nonlinear_earth_r(const double u_in[12], double Theta,
  double Phi, double Psi, double p, double q, double r, double K_p_T, double
  K_p_M, double m, double I_xx, double I_yy, double I_zz, double l_1, double l_2,
  double l_3, double l_4, double l_z, double Cl_alpha, double Cd_zero, double
  K_Cd, double Cm_alpha, double Cm_zero, double rho, double V, double S, double
  wing_chord, double flight_path_angle, double Beta, double computed_acc[6]);
static void c_compute_cost_and_gradient_fcn(const captured_var *W_act_motor,
  const captured_var *gamma_quadratic_du, const captured_var
  *desired_motor_value, const captured_var *gain_motor, const captured_var
  *W_dv_2, const captured_var *gamma_quadratic_dv, const b_captured_var
  *dv_global, const captured_var *S, const captured_var *V, const captured_var
  *rho, const captured_var *Beta, const captured_var *Theta, const captured_var *
  flight_path_angle, const captured_var *K_Cd, const captured_var *Cl_alpha,
  const captured_var *Cd_zero, const captured_var *Psi, const captured_var *Phi,
  const captured_var *K_p_T, const captured_var *gain_el, const captured_var
  *gain_az, const captured_var *m, const captured_var *W_act_tilt_el, const
  captured_var *desired_el_value, const captured_var *W_act_tilt_az, const
  captured_var *desired_az_value, const captured_var *W_dv_3, const captured_var
  *W_dv_1, const captured_var *W_dv_5, const captured_var *I_zz, const
  captured_var *p, const captured_var *r, const captured_var *I_xx, const
  captured_var *I_yy, const captured_var *l_z, const captured_var *K_p_M, const
  captured_var *Cm_zero, const captured_var *wing_chord, const captured_var *l_4,
  const captured_var *l_3, const captured_var *Cm_alpha, const captured_var
  *W_dv_6, const captured_var *q, const captured_var *l_1, const captured_var
  *l_2, const captured_var *W_dv_4, const double u_in[12], double *cost, double
  computed_gradient[12]);
static void c_xaxpy(int n, double a, const double x[12], int ix0, double y[72],
                    int iy0);
static double c_xnrm2(int n, const double x_data[], int ix0);
static double computeComplError(const double xCurrent[12], const int finiteLB[13],
  int mLB, const double lb[12], const int finiteUB[13], int mUB, const double
  ub[12], const double lambda[25], int iL0);
static double computeFval(const struct_T *obj, double workspace[325], const
  double H[144], const double f[13], const double x[13]);
static double computeFval_ReuseHx(const struct_T *obj, double workspace[325],
  const double f[13], const double x[13]);
static void computeGradLag(double workspace[13], int nVar, const double grad[13],
  const int finiteFixed[13], int mFixed, const int finiteLB[13], int mLB, const
  int finiteUB[13], int mUB, const double lambda[25]);
static void computeGrad_StoreHx(struct_T *obj, const double H[144], const double
  f[13], const double x[13]);
static void computeQ_(f_struct_T *obj, int nrows);
static void compute_deltax(const double H[144], i_struct_T *solution, e_struct_T
  *memspace, const f_struct_T *qrmanager, g_struct_T *cholmanager, const
  struct_T *objective, bool alwaysPositiveDef);
static void countsort(int x[25], int xLen, int workspace[25], int xMin, int xMax);
static void d_xaxpy(int n, double a, int ix0, double y[36], int iy0);
static double d_xnrm2(int n, const double x[72], int ix0);
static void deleteColMoveEnd(f_struct_T *obj, int idx);
static int div_nde_s32_floor(int numerator, int denominator);
static void driver(const double H[144], const double f[13], i_struct_T *solution,
                   e_struct_T *memspace, k_struct_T *workingset, f_struct_T
                   *qrmanager, g_struct_T *cholmanager, struct_T *objective,
                   h_struct_T *options, int runTimeOptions_MaxIterations);
static double e_xnrm2(int n, const double x[6], int ix0);
static void evalObjAndConstr(const d_struct_T *obj_objfun_workspace, const
  double x[12], double *fval, int *status);
static void evalObjAndConstrAndDerivatives(const d_struct_T
  *obj_objfun_workspace, const double x[12], double grad_workspace[13], double
  *fval, int *status);
static void factorQR(f_struct_T *obj, const double A[325], int mrows, int ncols);
static void factoryConstruct(d_struct_T *objfun_workspace, const double lb[12],
  const double ub[12], j_struct_T *obj);
static bool feasibleX0ForWorkingSet(double workspace[325], double xCurrent[13],
  const k_struct_T *workingset, f_struct_T *qrmanager);
static void feasibleratiotest(const double solution_xstar[13], const double
  solution_searchDir[13], int workingset_nVar, const double workingset_lb[13],
  const double workingset_ub[13], const int workingset_indexLB[13], const int
  workingset_indexUB[13], const int workingset_sizes[5], const int
  workingset_isActiveIdx[6], const bool workingset_isActiveConstr[25], const int
  workingset_nWConstr[5], bool isPhaseOne, double *alpha, bool *newBlocking, int
  *constrType, int *constrIdx);
static void fmincon(d_struct_T *fun_workspace, const double x0[12], const double
                    lb[12], const double ub[12], double x[12], double *fval,
                    double *exitflag, double *output_iterations, double
                    *output_funcCount, char output_algorithm[3], double
                    *output_constrviolation, double *output_stepsize, double
                    *output_lssteplength, double *output_firstorderopt);
static void fullColLDL2_(g_struct_T *obj, int NColsRemain);
static void iterate(const double H[144], const double f[13], i_struct_T
                    *solution, e_struct_T *memspace, k_struct_T *workingset,
                    f_struct_T *qrmanager, g_struct_T *cholmanager, struct_T
                    *objective, const char options_SolverName[7], double
                    options_StepTolerance, double options_ObjectiveLimit, int
                    runTimeOptions_MaxIterations);
static void linearForm_(bool obj_hasLinear, int obj_nvar, double workspace[325],
  const double H[144], const double f[13], const double x[13]);
static double maxConstraintViolation(const k_struct_T *obj, const double x[325],
  int ix0);
static void minimum(const double x[12], double *ex, int *idx);
static void mldivide(const double A_data[], const int A_size[2], const double B
                     [18], double Y_data[], int *Y_size);
static void qrf(double A[625], int m, int n, int nfxd, double tau[25]);
static double rt_hypotd_snf(double u0, double u1);
static void setProblemType(k_struct_T *obj, int PROBLEM_TYPE);
static void solve(const g_struct_T *obj, double rhs[13]);
static void sortLambdaQP(double lambda[25], int WorkingSet_nActiveConstr, const
  int WorkingSet_sizes[5], const int WorkingSet_isActiveIdx[6], const int
  WorkingSet_Wid[25], const int WorkingSet_Wlocalidx[25], double workspace[325]);
static bool step(int *STEP_TYPE, double Hessian[144], const double lb[12], const
                 double ub[12], i_struct_T *TrialState, b_struct_T
                 *MeritFunction, e_struct_T *memspace, k_struct_T *WorkingSet,
                 f_struct_T *QRManager, g_struct_T *CholManager, struct_T
                 *QPObjective, h_struct_T *qpoptions);
static void svd(const double A[72], double U[72], double s[6], double V[36]);
static void test_exit(b_struct_T *MeritFunction, const k_struct_T *WorkingSet,
                      i_struct_T *TrialState, const double lb[12], const double
                      ub[12], bool *Flags_gradOK, bool *Flags_fevalOK, bool
                      *Flags_done, bool *Flags_stepAccepted, bool
                      *Flags_failedLineSearch, int *Flags_stepType);
static void tic(void);
static void timeKeeper(double newTime_tv_sec, double newTime_tv_nsec);
static double toc(void);
static void xaxpy(int n, double a, int ix0, double y[72], int iy0);
static double xdotc(int n, const double x[72], int ix0, const double y[72], int
                    iy0);
static void xgemm(int m, int n, int k, const double A[144], int lda, const
                  double B[625], int ib0, double C[325]);
static void xgemv(int m, int n, const double A[625], const double x[13], double
                  y[325]);
static void xgeqp3(double A[625], int m, int n, int jpvt[25], double tau[25]);
static double xnrm2(int n, const double x[625], int ix0);
static int xpotrf(int n, double A[625]);
static void xrot(double x[36], int ix0, int iy0, double c, double s);
static void xrotg(double *a, double *b, double *c, double *s);
static void xswap(double x[36], int ix0, int iy0);
static void xzlarf(int m, int n, int iv0, double tau, double C[625], int ic0,
                   double work[25]);
static double xzlarfg(int n, double *alpha1, double x[625], int ix0);

/* Function Definitions */
/*
 * Arguments    : int nvar
 *                double Bk[144]
 *                const double sk[13]
 *                double yk[13]
 *                double workspace[325]
 * Return Type  : bool
 */
static bool BFGSUpdate(int nvar, double Bk[144], const double sk[13], double yk
  [13], double workspace[325])
{
  double curvatureS;
  double dotSY;
  double theta;
  int b_k;
  int i;
  int i1;
  int ia;
  int iac;
  int ix;
  int k;
  bool success;
  dotSY = 0.0;
  for (k = 0; k < nvar; k++) {
    dotSY += sk[k] * yk[k];
    workspace[k] = 0.0;
  }

  ix = 0;
  i = 12 * (nvar - 1) + 1;
  for (iac = 1; iac <= i; iac += 12) {
    i1 = (iac + nvar) - 1;
    for (ia = iac; ia <= i1; ia++) {
      k = ia - iac;
      workspace[k] += Bk[ia - 1] * sk[ix];
    }

    ix++;
  }

  curvatureS = 0.0;
  if (nvar >= 1) {
    for (k = 0; k < nvar; k++) {
      curvatureS += sk[k] * workspace[k];
    }
  }

  if (dotSY < 0.2 * curvatureS) {
    theta = 0.8 * curvatureS / (curvatureS - dotSY);
    if (nvar < 400) {
      for (b_k = 0; b_k < nvar; b_k++) {
        yk[b_k] *= theta;
      }
    } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

      for (b_k = 0; b_k < nvar; b_k++) {
        yk[b_k] *= theta;
      }
    }

    if (!(1.0 - theta == 0.0)) {
      ix = nvar - 1;
      for (k = 0; k <= ix; k++) {
        yk[k] += (1.0 - theta) * workspace[k];
      }
    }

    dotSY = 0.0;
    for (k = 0; k < nvar; k++) {
      dotSY += sk[k] * yk[k];
    }
  }

  if ((curvatureS > 2.2204460492503131E-16) && (dotSY > 2.2204460492503131E-16))
  {
    success = true;
  } else {
    success = false;
  }

  if (success) {
    theta = -1.0 / curvatureS;
    if (!(theta == 0.0)) {
      ix = 0;
      for (k = 0; k < nvar; k++) {
        if (workspace[k] != 0.0) {
          curvatureS = workspace[k] * theta;
          i = ix + 1;
          i1 = nvar + ix;
          for (iac = i; iac <= i1; iac++) {
            Bk[iac - 1] += workspace[(iac - ix) - 1] * curvatureS;
          }
        }

        ix += 12;
      }
    }

    theta = 1.0 / dotSY;
    if (!(theta == 0.0)) {
      ix = 0;
      for (k = 0; k < nvar; k++) {
        if (yk[k] != 0.0) {
          curvatureS = yk[k] * theta;
          i = ix + 1;
          i1 = nvar + ix;
          for (iac = i; iac <= i1; iac++) {
            Bk[iac - 1] += yk[(iac - ix) - 1] * curvatureS;
          }
        }

        ix += 12;
      }
    }
  }

  return success;
}

/*
 * Arguments    : i_struct_T *solution
 *                e_struct_T *memspace
 *                k_struct_T *workingset
 *                f_struct_T *qrmanager
 * Return Type  : void
 */
static void PresolveWorkingSet(i_struct_T *solution, e_struct_T *memspace,
  k_struct_T *workingset, f_struct_T *qrmanager)
{
  double tol;
  int idx;
  int idxDiag;
  int idx_col;
  int ix;
  int k;
  int mTotalWorkingEq;
  int mWorkingFixed;
  int nDepInd;
  int nVar;
  solution->state = 82;
  nVar = workingset->nVar - 1;
  mWorkingFixed = workingset->nWConstr[0];
  mTotalWorkingEq = workingset->nWConstr[0] + workingset->nWConstr[1];
  nDepInd = 0;
  if (mTotalWorkingEq > 0) {
    for (idxDiag = 0; idxDiag < mTotalWorkingEq; idxDiag++) {
      for (idx_col = 0; idx_col <= nVar; idx_col++) {
        qrmanager->QR[idxDiag + 25 * idx_col] = workingset->ATwset[idx_col + 13 *
          idxDiag];
      }
    }

    nDepInd = mTotalWorkingEq - workingset->nVar;
    if (nDepInd <= 0) {
      nDepInd = 0;
    }

    if (nVar + 1 < 400) {
      if (nVar >= 0) {
        memset(&qrmanager->jpvt[0], 0, (nVar + 1) * sizeof(int));
      }
    } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

      for (idx = 0; idx <= nVar; idx++) {
        qrmanager->jpvt[idx] = 0;
      }
    }

    if (mTotalWorkingEq * workingset->nVar == 0) {
      qrmanager->mrows = mTotalWorkingEq;
      qrmanager->ncols = workingset->nVar;
      qrmanager->minRowCol = 0;
    } else {
      qrmanager->usedPivoting = true;
      qrmanager->mrows = mTotalWorkingEq;
      qrmanager->ncols = workingset->nVar;
      idxDiag = workingset->nVar;
      if (mTotalWorkingEq <= idxDiag) {
        idxDiag = mTotalWorkingEq;
      }

      qrmanager->minRowCol = idxDiag;
      xgeqp3(qrmanager->QR, mTotalWorkingEq, workingset->nVar, qrmanager->jpvt,
             qrmanager->tau);
    }

    tol = 100.0 * (double)workingset->nVar * 2.2204460492503131E-16;
    idxDiag = workingset->nVar;
    if (idxDiag > mTotalWorkingEq) {
      idxDiag = mTotalWorkingEq;
    }

    idxDiag += 25 * (idxDiag - 1);
    while ((idxDiag > 0) && (fabs(qrmanager->QR[idxDiag - 1]) < tol)) {
      idxDiag -= 26;
      nDepInd++;
    }

    if (nDepInd > 0) {
      bool exitg1;
      computeQ_(qrmanager, qrmanager->mrows);
      idx_col = 0;
      exitg1 = false;
      while ((!exitg1) && (idx_col <= nDepInd - 1)) {
        double qtb;
        ix = 25 * ((mTotalWorkingEq - idx_col) - 1);
        qtb = 0.0;
        for (k = 0; k < mTotalWorkingEq; k++) {
          qtb += qrmanager->Q[ix + k] * workingset->bwset[k];
        }

        if (fabs(qtb) >= tol) {
          nDepInd = -1;
          exitg1 = true;
        } else {
          idx_col++;
        }
      }
    }

    if (nDepInd > 0) {
      for (idx_col = 0; idx_col < mTotalWorkingEq; idx_col++) {
        idxDiag = 25 * idx_col;
        ix = 13 * idx_col;
        for (k = 0; k <= nVar; k++) {
          qrmanager->QR[idxDiag + k] = workingset->ATwset[ix + k];
        }
      }

      if (mWorkingFixed < 400) {
        for (idx = 0; idx < mWorkingFixed; idx++) {
          qrmanager->jpvt[idx] = 1;
        }
      } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

        for (idx = 0; idx < mWorkingFixed; idx++) {
          qrmanager->jpvt[idx] = 1;
        }
      }

      ix = workingset->nWConstr[0] + 1;
      if ((mTotalWorkingEq - ix) + 1 < 400) {
        if (ix <= mTotalWorkingEq) {
          memset(&qrmanager->jpvt[ix + -1], 0, ((mTotalWorkingEq - ix) + 1) *
                 sizeof(int));
        }
      } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

        for (idx = ix; idx <= mTotalWorkingEq; idx++) {
          qrmanager->jpvt[idx - 1] = 0;
        }
      }

      if (workingset->nVar * mTotalWorkingEq == 0) {
        qrmanager->mrows = workingset->nVar;
        qrmanager->ncols = mTotalWorkingEq;
        qrmanager->minRowCol = 0;
      } else {
        qrmanager->usedPivoting = true;
        qrmanager->mrows = workingset->nVar;
        qrmanager->ncols = mTotalWorkingEq;
        idxDiag = workingset->nVar;
        if (idxDiag > mTotalWorkingEq) {
          idxDiag = mTotalWorkingEq;
        }

        qrmanager->minRowCol = idxDiag;
        xgeqp3(qrmanager->QR, workingset->nVar, mTotalWorkingEq, qrmanager->jpvt,
               qrmanager->tau);
      }

      if (nDepInd < 400) {
        for (idx = 0; idx < nDepInd; idx++) {
          memspace->workspace_int[idx] = qrmanager->jpvt[(mTotalWorkingEq -
            nDepInd) + idx];
        }
      } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

        for (idx = 0; idx < nDepInd; idx++) {
          memspace->workspace_int[idx] = qrmanager->jpvt[(mTotalWorkingEq -
            nDepInd) + idx];
        }
      }

      countsort(memspace->workspace_int, nDepInd, memspace->workspace_sort, 1,
                mTotalWorkingEq);
      for (idx_col = nDepInd; idx_col >= 1; idx_col--) {
        ix = workingset->nWConstr[0] + workingset->nWConstr[1];
        if (ix != 0) {
          idxDiag = memspace->workspace_int[idx_col - 1];
          if (idxDiag <= ix) {
            if ((workingset->nActiveConstr == ix) || (idxDiag == ix)) {
              workingset->mEqRemoved++;

              /* A check that is always false is detected at compile-time. Eliminating code that follows. */
            } else {
              workingset->mEqRemoved++;

              /* A check that is always false is detected at compile-time. Eliminating code that follows. */
            }
          }
        }
      }
    }
  }

  if ((nDepInd != -1) && (workingset->nActiveConstr <= 25)) {
    bool guard1 = false;
    bool okWorkingSet;
    RemoveDependentIneq_(workingset, qrmanager, memspace, 100.0);
    okWorkingSet = feasibleX0ForWorkingSet(memspace->workspace_double,
      solution->xstar, workingset, qrmanager);
    guard1 = false;
    if (!okWorkingSet) {
      RemoveDependentIneq_(workingset, qrmanager, memspace, 1000.0);
      okWorkingSet = feasibleX0ForWorkingSet(memspace->workspace_double,
        solution->xstar, workingset, qrmanager);
      if (!okWorkingSet) {
        solution->state = -7;
      } else {
        guard1 = true;
      }
    } else {
      guard1 = true;
    }

    if (guard1 && (workingset->nWConstr[0] + workingset->nWConstr[1] ==
                   workingset->nVar)) {
      tol = b_maxConstraintViolation(workingset, solution->xstar);
      if (tol > 1.0E-6) {
        solution->state = -2;
      }
    }
  } else {
    solution->state = -3;
    idxDiag = (workingset->nWConstr[0] + workingset->nWConstr[1]) + 1;
    ix = workingset->nActiveConstr;
    for (idx_col = idxDiag; idx_col <= ix; idx_col++) {
      workingset->isActiveConstr[(workingset->isActiveIdx[workingset->
        Wid[idx_col - 1] - 1] + workingset->Wlocalidx[idx_col - 1]) - 2] = false;
    }

    workingset->nWConstr[2] = 0;
    workingset->nWConstr[3] = 0;
    workingset->nWConstr[4] = 0;
    workingset->nActiveConstr = workingset->nWConstr[0] + workingset->nWConstr[1];
  }
}

/*
 * Arguments    : k_struct_T *workingset
 *                f_struct_T *qrmanager
 *                e_struct_T *memspace
 *                double tolfactor
 * Return Type  : void
 */
static void RemoveDependentIneq_(k_struct_T *workingset, f_struct_T *qrmanager,
  e_struct_T *memspace, double tolfactor)
{
  int b_idx;
  int idx;
  int k;
  int nActiveConstr;
  int nDepIneq;
  int nFixedConstr;
  int nVar;
  nActiveConstr = workingset->nActiveConstr;
  nFixedConstr = workingset->nWConstr[0] + workingset->nWConstr[1];
  nVar = workingset->nVar;
  if ((workingset->nWConstr[2] + workingset->nWConstr[3]) + workingset->
      nWConstr[4] > 0) {
    double tol;
    int i;
    int idxDiag;
    tol = tolfactor * (double)workingset->nVar * 2.2204460492503131E-16;
    if (nFixedConstr < 400) {
      for (idx = 0; idx < nFixedConstr; idx++) {
        qrmanager->jpvt[idx] = 1;
      }
    } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

      for (idx = 0; idx < nFixedConstr; idx++) {
        qrmanager->jpvt[idx] = 1;
      }
    }

    i = nFixedConstr + 1;
    if (nActiveConstr - nFixedConstr < 400) {
      if (i <= nActiveConstr) {
        memset(&qrmanager->jpvt[i + -1], 0, ((nActiveConstr - i) + 1) * sizeof
               (int));
      }
    } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

      for (idx = i; idx <= nActiveConstr; idx++) {
        qrmanager->jpvt[idx - 1] = 0;
      }
    }

    i = workingset->nActiveConstr;
    for (nDepIneq = 0; nDepIneq < i; nDepIneq++) {
      idxDiag = 25 * nDepIneq;
      nActiveConstr = 13 * nDepIneq;
      for (k = 0; k < nVar; k++) {
        qrmanager->QR[idxDiag + k] = workingset->ATwset[nActiveConstr + k];
      }
    }

    if (workingset->nVar * workingset->nActiveConstr == 0) {
      qrmanager->mrows = workingset->nVar;
      qrmanager->ncols = workingset->nActiveConstr;
      qrmanager->minRowCol = 0;
    } else {
      qrmanager->usedPivoting = true;
      qrmanager->mrows = workingset->nVar;
      qrmanager->ncols = workingset->nActiveConstr;
      idxDiag = workingset->nVar;
      nActiveConstr = workingset->nActiveConstr;
      if (idxDiag <= nActiveConstr) {
        nActiveConstr = idxDiag;
      }

      qrmanager->minRowCol = nActiveConstr;
      xgeqp3(qrmanager->QR, workingset->nVar, workingset->nActiveConstr,
             qrmanager->jpvt, qrmanager->tau);
    }

    nDepIneq = 0;
    for (b_idx = workingset->nActiveConstr - 1; b_idx + 1 > nVar; b_idx--) {
      nDepIneq++;
      memspace->workspace_int[nDepIneq - 1] = qrmanager->jpvt[b_idx];
    }

    if (b_idx + 1 <= workingset->nVar) {
      idxDiag = b_idx + 25 * b_idx;
      while ((b_idx + 1 > nFixedConstr) && (fabs(qrmanager->QR[idxDiag]) < tol))
      {
        nDepIneq++;
        memspace->workspace_int[nDepIneq - 1] = qrmanager->jpvt[b_idx];
        b_idx--;
        idxDiag -= 26;
      }
    }

    countsort(memspace->workspace_int, nDepIneq, memspace->workspace_sort,
              nFixedConstr + 1, workingset->nActiveConstr);
    for (b_idx = nDepIneq; b_idx >= 1; b_idx--) {
      idxDiag = memspace->workspace_int[b_idx - 1] - 1;
      nActiveConstr = workingset->Wid[idxDiag] - 1;
      workingset->isActiveConstr[(workingset->isActiveIdx[nActiveConstr] +
        workingset->Wlocalidx[idxDiag]) - 2] = false;
      workingset->Wid[idxDiag] = workingset->Wid[workingset->nActiveConstr - 1];
      workingset->Wlocalidx[idxDiag] = workingset->Wlocalidx
        [workingset->nActiveConstr - 1];
      i = workingset->nVar;
      for (k = 0; k < i; k++) {
        workingset->ATwset[k + 13 * idxDiag] = workingset->ATwset[k + 13 *
          (workingset->nActiveConstr - 1)];
      }

      workingset->bwset[idxDiag] = workingset->bwset[workingset->nActiveConstr -
        1];
      workingset->nActiveConstr--;
      workingset->nWConstr[nActiveConstr]--;
    }
  }
}

/*
 * Arguments    : k_struct_T *obj
 *                int TYPE
 *                int idx_local
 * Return Type  : void
 */
static void addBoundToActiveSetMatrix_(k_struct_T *obj, int TYPE, int idx_local)
{
  int colOffset;
  int i;
  int i1;
  int idx;
  int idx_bnd_local;
  obj->nWConstr[TYPE - 1]++;
  obj->isActiveConstr[(obj->isActiveIdx[TYPE - 1] + idx_local) - 2] = true;
  obj->nActiveConstr++;
  obj->Wid[obj->nActiveConstr - 1] = TYPE;
  obj->Wlocalidx[obj->nActiveConstr - 1] = idx_local;
  colOffset = 13 * (obj->nActiveConstr - 1) - 1;
  if (TYPE == 5) {
    idx_bnd_local = obj->indexUB[idx_local - 1];
    obj->bwset[obj->nActiveConstr - 1] = obj->ub[idx_bnd_local - 1];
  } else {
    idx_bnd_local = obj->indexLB[idx_local - 1];
    obj->bwset[obj->nActiveConstr - 1] = obj->lb[idx_bnd_local - 1];
  }

  if (idx_bnd_local - 2 >= 0) {
    memset(&obj->ATwset[colOffset + 1], 0, (((idx_bnd_local + colOffset) -
             colOffset) - 1) * sizeof(double));
  }

  obj->ATwset[idx_bnd_local + colOffset] = 2.0 * (double)(TYPE == 5) - 1.0;
  i = idx_bnd_local + 1;
  i1 = obj->nVar;
  if (i1 - idx_bnd_local < 400) {
    if (i <= i1) {
      memset(&obj->ATwset[i + colOffset], 0, ((((i1 + colOffset) - i) -
               colOffset) + 1) * sizeof(double));
    }
  } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

    for (idx = i; idx <= i1; idx++) {
      obj->ATwset[idx + colOffset] = 0.0;
    }
  }

  switch (obj->probType) {
   case 3:
   case 2:
    break;

   default:
    obj->ATwset[obj->nVar + colOffset] = -1.0;
    break;
  }
}

/*
 * Arguments    : double workspace[325]
 *                int nVar
 *                const double grad[13]
 *                const int finiteFixed[13]
 *                int mFixed
 *                const int finiteLB[13]
 *                int mLB
 *                const int finiteUB[13]
 *                int mUB
 *                const double lambda[25]
 * Return Type  : void
 */
static void b_computeGradLag(double workspace[325], int nVar, const double grad
  [13], const int finiteFixed[13], int mFixed, const int finiteLB[13], int mLB,
  const int finiteUB[13], int mUB, const double lambda[25])
{
  int b_i;
  int i;
  int iL0;
  int idx;
  if (nVar < 400) {
    if (nVar - 1 >= 0) {
      memcpy(&workspace[0], &grad[0], nVar * sizeof(double));
    }
  } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

    for (i = 0; i < nVar; i++) {
      workspace[i] = grad[i];
    }
  }

  for (idx = 0; idx < mFixed; idx++) {
    b_i = finiteFixed[idx];
    workspace[b_i - 1] += lambda[idx];
  }

  for (idx = 0; idx < mLB; idx++) {
    b_i = finiteLB[idx];
    workspace[b_i - 1] -= lambda[mFixed + idx];
  }

  iL0 = mFixed + mLB;
  for (idx = 0; idx < mUB; idx++) {
    b_i = finiteUB[idx];
    workspace[b_i - 1] += lambda[iL0 + idx];
  }
}

/*
 * Arguments    : const double lb[12]
 *                const double ub[12]
 *                i_struct_T *TrialState
 *                b_struct_T *MeritFunction
 *                const l_struct_T *FcnEvaluator
 *                e_struct_T *memspace
 *                k_struct_T *WorkingSet
 *                double Hessian[144]
 *                f_struct_T *QRManager
 *                g_struct_T *CholManager
 *                struct_T *QPObjective
 * Return Type  : void
 */
static void b_driver(const double lb[12], const double ub[12], i_struct_T
                     *TrialState, b_struct_T *MeritFunction, const l_struct_T
                     *FcnEvaluator, e_struct_T *memspace, k_struct_T *WorkingSet,
                     double Hessian[144], f_struct_T *QRManager, g_struct_T
                     *CholManager, struct_T *QPObjective)
{
  static const signed char iv[144] = { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
  };

  static const char qpoptions_SolverName[7] = { 'f', 'm', 'i', 'n', 'c', 'o',
    'n' };

  c_struct_T Flags;
  h_struct_T b_expl_temp;
  h_struct_T expl_temp;
  int i;
  int i1;
  int ineqStart;
  int mConstr;
  int mFixed;
  int mLB;
  int mUB;
  int nVar_tmp_tmp;
  int qpoptions_MaxIterations;
  int u1;
  memset(&QPObjective->grad[0], 0, 13U * sizeof(double));
  memset(&QPObjective->Hx[0], 0, 12U * sizeof(double));
  QPObjective->hasLinear = true;
  QPObjective->nvar = 12;
  QPObjective->maxVar = 13;
  QPObjective->beta = 0.0;
  QPObjective->rho = 0.0;
  QPObjective->objtype = 3;
  QPObjective->prev_objtype = 3;
  QPObjective->prev_nvar = 0;
  QPObjective->prev_hasLinear = false;
  QPObjective->gammaScalar = 0.0;
  CholManager->ldm = 25;
  CholManager->ndims = 0;
  CholManager->info = 0;
  CholManager->scaleFactor = 0.0;
  CholManager->ConvexCheck = true;
  CholManager->regTol_ = rtInf;
  CholManager->workspace_ = rtInf;
  CholManager->workspace2_ = rtInf;
  QRManager->ldq = 25;

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

  for (i = 0; i < 625; i++) {
    CholManager->FMat[i] = 0.0;
    QRManager->QR[i] = 0.0;
    QRManager->Q[i] = 0.0;
  }

  QRManager->mrows = 0;
  QRManager->ncols = 0;
  memset(&QRManager->jpvt[0], 0, 25U * sizeof(int));
  memset(&QRManager->tau[0], 0, 25U * sizeof(double));
  QRManager->minRowCol = 0;
  QRManager->usedPivoting = false;
  for (i1 = 0; i1 < 144; i1++) {
    Hessian[i1] = iv[i1];
  }

  nVar_tmp_tmp = WorkingSet->nVar - 1;
  mFixed = WorkingSet->sizes[0];
  mLB = WorkingSet->sizes[3];
  mUB = WorkingSet->sizes[4];
  mConstr = (WorkingSet->sizes[0] + WorkingSet->sizes[3]) + WorkingSet->sizes[4];
  ineqStart = WorkingSet->nVar;
  u1 = (WorkingSet->sizes[3] + WorkingSet->sizes[4]) + (WorkingSet->sizes[0] <<
    1);
  if (ineqStart >= u1) {
    u1 = ineqStart;
  }

  qpoptions_MaxIterations = 10 * u1;
  TrialState->steplength = 1.0;
  test_exit(MeritFunction, WorkingSet, TrialState, lb, ub, &Flags.gradOK,
            &Flags.fevalOK, &Flags.done, &Flags.stepAccepted,
            &Flags.failedLineSearch, &Flags.stepType);
  TrialState->sqpFval_old = TrialState->sqpFval;
  for (u1 = 0; u1 < 12; u1++) {
    TrialState->xstarsqp_old[u1] = TrialState->xstarsqp[u1];
    TrialState->grad_old[u1] = TrialState->grad[u1];
  }

  if (!Flags.done) {
    TrialState->sqpIterations = 1;
  }

  while (!Flags.done) {
    while (!(Flags.stepAccepted || Flags.failedLineSearch)) {
      if (Flags.stepType != 3) {
        for (u1 = 0; u1 < mLB; u1++) {
          WorkingSet->lb[WorkingSet->indexLB[u1] - 1] = -lb[WorkingSet->
            indexLB[u1] - 1] + TrialState->xstarsqp[WorkingSet->indexLB[u1] - 1];
        }

        for (u1 = 0; u1 < mUB; u1++) {
          WorkingSet->ub[WorkingSet->indexUB[u1] - 1] = ub[WorkingSet->
            indexUB[u1] - 1] - TrialState->xstarsqp[WorkingSet->indexUB[u1] - 1];
        }

        for (u1 = 0; u1 < mFixed; u1++) {
          WorkingSet->ub[WorkingSet->indexFixed[u1] - 1] = ub
            [WorkingSet->indexFixed[u1] - 1] - TrialState->xstarsqp
            [WorkingSet->indexFixed[u1] - 1];
          WorkingSet->bwset[u1] = ub[WorkingSet->indexFixed[u1] - 1] -
            TrialState->xstarsqp[WorkingSet->indexFixed[u1] - 1];
        }

        if (WorkingSet->nActiveConstr > mFixed) {
          ineqStart = mFixed + 1;
          if (ineqStart < 1) {
            ineqStart = 1;
          }

          i1 = WorkingSet->nActiveConstr;
          for (u1 = ineqStart; u1 <= i1; u1++) {
            switch (WorkingSet->Wid[u1 - 1]) {
             case 4:
              WorkingSet->bwset[u1 - 1] = WorkingSet->lb[WorkingSet->
                indexLB[WorkingSet->Wlocalidx[u1 - 1] - 1] - 1];
              break;

             case 5:
              WorkingSet->bwset[u1 - 1] = WorkingSet->ub[WorkingSet->
                indexUB[WorkingSet->Wlocalidx[u1 - 1] - 1] - 1];
              break;

             default:
              /* A check that is always false is detected at compile-time. Eliminating code that follows. */
              break;
            }
          }
        }
      }

      expl_temp.IterDisplayQP = false;
      expl_temp.RemainFeasible = false;
      expl_temp.ProbRelTolFactor = 1.0;
      expl_temp.ConstrRelTolFactor = 1.0;
      expl_temp.PricingTolerance = 0.0;
      expl_temp.ObjectiveLimit = rtMinusInf;
      expl_temp.ConstraintTolerance = 1.0E-6;
      expl_temp.OptimalityTolerance = 2.2204460492503131E-14;
      expl_temp.StepTolerance = 1.0E-6;
      expl_temp.MaxIterations = qpoptions_MaxIterations;
      for (i1 = 0; i1 < 7; i1++) {
        expl_temp.SolverName[i1] = qpoptions_SolverName[i1];
      }

      b_expl_temp = expl_temp;
      Flags.stepAccepted = step(&Flags.stepType, Hessian, lb, ub, TrialState,
        MeritFunction, memspace, WorkingSet, QRManager, CholManager, QPObjective,
        &b_expl_temp);
      if (Flags.stepAccepted) {
        for (u1 = 0; u1 <= nVar_tmp_tmp; u1++) {
          TrialState->xstarsqp[u1] += TrialState->delta_x[u1];
        }

        evalObjAndConstr(&FcnEvaluator->objfun.workspace, TrialState->xstarsqp,
                         &TrialState->sqpFval, &ineqStart);
        Flags.fevalOK = (ineqStart == 1);
        TrialState->FunctionEvaluations++;
        if (Flags.fevalOK) {
          MeritFunction->phiFullStep = TrialState->sqpFval;
        } else {
          MeritFunction->phiFullStep = rtInf;
        }
      }

      if ((Flags.stepType == 1) && Flags.stepAccepted && Flags.fevalOK &&
          (MeritFunction->phi < MeritFunction->phiFullStep) &&
          (TrialState->sqpFval < TrialState->sqpFval_old)) {
        Flags.stepType = 3;
        Flags.stepAccepted = false;
      } else {
        double alpha;
        double phi_alpha;
        bool evalWellDefined;
        bool socTaken;
        if ((Flags.stepType == 3) && Flags.stepAccepted) {
          socTaken = true;
        } else {
          socTaken = false;
        }

        evalWellDefined = Flags.fevalOK;
        i1 = WorkingSet->nVar - 1;
        alpha = 1.0;
        ineqStart = 1;
        phi_alpha = MeritFunction->phiFullStep;
        if (i1 >= 0) {
          memcpy(&TrialState->searchDir[0], &TrialState->delta_x[0], (i1 + 1) *
                 sizeof(double));
        }

        int exitg1;
        do {
          exitg1 = 0;
          if (TrialState->FunctionEvaluations < 120) {
            if (evalWellDefined && (phi_alpha <= MeritFunction->phi + alpha *
                                    0.0001 * MeritFunction->phiPrimePlus)) {
              exitg1 = 1;
            } else {
              bool exitg2;
              bool tooSmallX;
              alpha *= 0.7;
              for (u1 = 0; u1 <= i1; u1++) {
                TrialState->delta_x[u1] = alpha * TrialState->xstar[u1];
              }

              if (socTaken) {
                phi_alpha = alpha * alpha;
                if ((i1 + 1 >= 1) && (!(phi_alpha == 0.0))) {
                  for (u1 = 0; u1 <= i1; u1++) {
                    TrialState->delta_x[u1] += phi_alpha *
                      TrialState->socDirection[u1];
                  }
                }
              }

              tooSmallX = true;
              u1 = 0;
              exitg2 = false;
              while ((!exitg2) && (u1 <= i1)) {
                if (1.0E-9 * fmax(1.0, fabs(TrialState->xstarsqp[u1])) <= fabs
                    (TrialState->delta_x[u1])) {
                  tooSmallX = false;
                  exitg2 = true;
                } else {
                  u1++;
                }
              }

              if (tooSmallX) {
                ineqStart = -2;
                exitg1 = 1;
              } else {
                for (u1 = 0; u1 <= i1; u1++) {
                  TrialState->xstarsqp[u1] = TrialState->xstarsqp_old[u1] +
                    TrialState->delta_x[u1];
                }

                evalObjAndConstr(&FcnEvaluator->objfun.workspace,
                                 TrialState->xstarsqp, &TrialState->sqpFval, &u1);
                TrialState->FunctionEvaluations++;
                evalWellDefined = (u1 == 1);
                if (evalWellDefined) {
                  phi_alpha = TrialState->sqpFval;
                } else {
                  phi_alpha = rtInf;
                }
              }
            }
          } else {
            ineqStart = 0;
            exitg1 = 1;
          }
        } while (exitg1 == 0);

        Flags.fevalOK = evalWellDefined;
        TrialState->steplength = alpha;
        if (ineqStart > 0) {
          Flags.stepAccepted = true;
        } else {
          Flags.failedLineSearch = true;
        }
      }
    }

    if (Flags.stepAccepted && (!Flags.failedLineSearch)) {
      for (u1 = 0; u1 <= nVar_tmp_tmp; u1++) {
        TrialState->xstarsqp[u1] = TrialState->xstarsqp_old[u1] +
          TrialState->delta_x[u1];
      }

      for (u1 = 0; u1 < mConstr; u1++) {
        TrialState->lambdasqp[u1] += TrialState->steplength *
          (TrialState->lambda[u1] - TrialState->lambdasqp[u1]);
      }

      TrialState->sqpFval_old = TrialState->sqpFval;
      for (u1 = 0; u1 < 12; u1++) {
        TrialState->xstarsqp_old[u1] = TrialState->xstarsqp[u1];
        TrialState->grad_old[u1] = TrialState->grad[u1];
      }

      Flags.gradOK = true;
      evalObjAndConstrAndDerivatives(&FcnEvaluator->objfun.workspace,
        TrialState->xstarsqp, TrialState->grad, &TrialState->sqpFval, &ineqStart);
      TrialState->FunctionEvaluations++;
      Flags.fevalOK = (ineqStart == 1);
    } else {
      TrialState->sqpFval = TrialState->sqpFval_old;
      memcpy(&TrialState->xstarsqp[0], &TrialState->xstarsqp_old[0], 12U *
             sizeof(double));
    }

    b_test_exit(&Flags, memspace, MeritFunction, WorkingSet, TrialState,
                QRManager, lb, ub);
    if ((!Flags.done) && Flags.stepAccepted) {
      Flags.stepAccepted = false;
      Flags.stepType = 1;
      Flags.failedLineSearch = false;
      if (nVar_tmp_tmp >= 0) {
        memcpy(&TrialState->delta_gradLag[0], &TrialState->grad[0],
               (nVar_tmp_tmp + 1) * sizeof(double));
      }

      if (nVar_tmp_tmp + 1 >= 1) {
        for (u1 = 0; u1 <= nVar_tmp_tmp; u1++) {
          TrialState->delta_gradLag[u1] += -TrialState->grad_old[u1];
        }
      }

      BFGSUpdate(nVar_tmp_tmp + 1, Hessian, TrialState->delta_x,
                 TrialState->delta_gradLag, memspace->workspace_double);
      TrialState->sqpIterations++;
    }
  }
}

/*
 * Arguments    : const k_struct_T *obj
 *                const double x[13]
 * Return Type  : double
 */
static double b_maxConstraintViolation(const k_struct_T *obj, const double x[13])
{
  double v;
  int idx;
  int mFixed;
  int mLB;
  int mUB;
  mLB = obj->sizes[3];
  mUB = obj->sizes[4];
  mFixed = obj->sizes[0];
  v = 0.0;
  if (obj->sizes[3] > 0) {
    for (idx = 0; idx < mLB; idx++) {
      int idxLB;
      idxLB = obj->indexLB[idx] - 1;
      v = fmax(v, -x[idxLB] - obj->lb[idxLB]);
    }
  }

  if (obj->sizes[4] > 0) {
    for (idx = 0; idx < mUB; idx++) {
      mLB = obj->indexUB[idx] - 1;
      v = fmax(v, x[mLB] - obj->ub[mLB]);
    }
  }

  if (obj->sizes[0] > 0) {
    for (idx = 0; idx < mFixed; idx++) {
      v = fmax(v, fabs(x[obj->indexFixed[idx] - 1] - obj->ub[obj->indexFixed[idx]
                       - 1]));
    }
  }

  return v;
}

/*
 * Arguments    : c_struct_T *Flags
 *                e_struct_T *memspace
 *                b_struct_T *MeritFunction
 *                const k_struct_T *WorkingSet
 *                i_struct_T *TrialState
 *                f_struct_T *QRManager
 *                const double lb[12]
 *                const double ub[12]
 * Return Type  : void
 */
static void b_test_exit(c_struct_T *Flags, e_struct_T *memspace, b_struct_T
  *MeritFunction, const k_struct_T *WorkingSet, i_struct_T *TrialState,
  f_struct_T *QRManager, const double lb[12], const double ub[12])
{
  double optimRelativeFactor;
  double s;
  double smax;
  int b_k;
  int idxFiniteLB;
  int idx_max;
  int k;
  int mFixed;
  int mLB;
  int mLambda;
  int mUB;
  int nVar;
  bool dxTooSmall;
  bool exitg1;
  bool isFeasible;
  nVar = WorkingSet->nVar;
  mFixed = WorkingSet->sizes[0];
  mLB = WorkingSet->sizes[3];
  mUB = WorkingSet->sizes[4];
  mLambda = ((WorkingSet->sizes[0] + WorkingSet->sizes[3]) + WorkingSet->sizes[4])
    - 1;
  if (mLambda + 1 < 400) {
    if (mLambda >= 0) {
      memcpy(&TrialState->lambdaStopTest[0], &TrialState->lambdasqp[0], (mLambda
              + 1) * sizeof(double));
    }
  } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

    for (k = 0; k <= mLambda; k++) {
      TrialState->lambdaStopTest[k] = TrialState->lambdasqp[k];
    }
  }

  computeGradLag(TrialState->gradLag, WorkingSet->nVar, TrialState->grad,
                 WorkingSet->indexFixed, WorkingSet->sizes[0],
                 WorkingSet->indexLB, WorkingSet->sizes[3], WorkingSet->indexUB,
                 WorkingSet->sizes[4], TrialState->lambdaStopTest);
  if (WorkingSet->nVar < 1) {
    idx_max = 0;
  } else {
    idx_max = 1;
    if (WorkingSet->nVar > 1) {
      smax = fabs(TrialState->grad[0]);
      for (b_k = 2; b_k <= nVar; b_k++) {
        s = fabs(TrialState->grad[b_k - 1]);
        if (s > smax) {
          idx_max = b_k;
          smax = s;
        }
      }
    }
  }

  optimRelativeFactor = fmax(1.0, fabs(TrialState->grad[idx_max - 1]));
  if (rtIsInf(optimRelativeFactor)) {
    optimRelativeFactor = 1.0;
  }

  smax = 0.0;
  for (idx_max = 0; idx_max < mLB; idx_max++) {
    idxFiniteLB = WorkingSet->indexLB[idx_max] - 1;
    smax = fmax(smax, lb[idxFiniteLB] - TrialState->xstarsqp[idxFiniteLB]);
  }

  for (idx_max = 0; idx_max < mUB; idx_max++) {
    idxFiniteLB = WorkingSet->indexUB[idx_max] - 1;
    smax = fmax(smax, TrialState->xstarsqp[idxFiniteLB] - ub[idxFiniteLB]);
  }

  MeritFunction->nlpPrimalFeasError = smax;
  if (TrialState->sqpIterations == 0) {
    MeritFunction->feasRelativeFactor = fmax(1.0, smax);
  }

  isFeasible = (smax <= 1.0E-6 * MeritFunction->feasRelativeFactor);
  dxTooSmall = true;
  smax = 0.0;
  idx_max = 0;
  exitg1 = false;
  while ((!exitg1) && (idx_max <= WorkingSet->nVar - 1)) {
    dxTooSmall = ((!rtIsInf(TrialState->gradLag[idx_max])) && (!rtIsNaN
      (TrialState->gradLag[idx_max])));
    if (!dxTooSmall) {
      exitg1 = true;
    } else {
      smax = fmax(smax, fabs(TrialState->gradLag[idx_max]));
      idx_max++;
    }
  }

  Flags->gradOK = dxTooSmall;
  MeritFunction->nlpDualFeasError = smax;
  if (!dxTooSmall) {
    Flags->done = true;
    if (isFeasible) {
      TrialState->sqpExitFlag = 2;
    } else {
      TrialState->sqpExitFlag = -2;
    }
  } else {
    MeritFunction->nlpComplError = computeComplError(TrialState->xstarsqp,
      WorkingSet->indexLB, WorkingSet->sizes[3], lb, WorkingSet->indexUB,
      WorkingSet->sizes[4], ub, TrialState->lambdaStopTest, WorkingSet->sizes[0]
      + 1);
    MeritFunction->firstOrderOpt = fmax(smax, MeritFunction->nlpComplError);
    if (TrialState->sqpIterations > 1) {
      double d;
      double nlpComplErrorTmp;
      b_computeGradLag(memspace->workspace_double, WorkingSet->nVar,
                       TrialState->grad, WorkingSet->indexFixed,
                       WorkingSet->sizes[0], WorkingSet->indexLB,
                       WorkingSet->sizes[3], WorkingSet->indexUB,
                       WorkingSet->sizes[4], TrialState->lambdaStopTestPrev);
      s = 0.0;
      idx_max = 0;
      while ((idx_max <= WorkingSet->nVar - 1) && ((!rtIsInf
               (memspace->workspace_double[idx_max])) && (!rtIsNaN
               (memspace->workspace_double[idx_max])))) {
        s = fmax(s, fabs(memspace->workspace_double[idx_max]));
        idx_max++;
      }

      nlpComplErrorTmp = computeComplError(TrialState->xstarsqp,
        WorkingSet->indexLB, WorkingSet->sizes[3], lb, WorkingSet->indexUB,
        WorkingSet->sizes[4], ub, TrialState->lambdaStopTestPrev,
        WorkingSet->sizes[0] + 1);
      d = fmax(s, nlpComplErrorTmp);
      if (d < fmax(smax, MeritFunction->nlpComplError)) {
        MeritFunction->nlpDualFeasError = s;
        MeritFunction->nlpComplError = nlpComplErrorTmp;
        MeritFunction->firstOrderOpt = d;
        if (mLambda + 1 < 400) {
          if (mLambda >= 0) {
            memcpy(&TrialState->lambdaStopTest[0],
                   &TrialState->lambdaStopTestPrev[0], (mLambda + 1) * sizeof
                   (double));
          }
        } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

          for (k = 0; k <= mLambda; k++) {
            TrialState->lambdaStopTest[k] = TrialState->lambdaStopTestPrev[k];
          }
        }
      } else if (mLambda + 1 < 400) {
        if (mLambda >= 0) {
          memcpy(&TrialState->lambdaStopTestPrev[0], &TrialState->
                 lambdaStopTest[0], (mLambda + 1) * sizeof(double));
        }
      } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

        for (k = 0; k <= mLambda; k++) {
          TrialState->lambdaStopTestPrev[k] = TrialState->lambdaStopTest[k];
        }
      }
    } else if (mLambda + 1 < 400) {
      if (mLambda >= 0) {
        memcpy(&TrialState->lambdaStopTestPrev[0], &TrialState->lambdaStopTest[0],
               (mLambda + 1) * sizeof(double));
      }
    } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

      for (k = 0; k <= mLambda; k++) {
        TrialState->lambdaStopTestPrev[k] = TrialState->lambdaStopTest[k];
      }
    }

    if (isFeasible && (MeritFunction->nlpDualFeasError <= 1.0E-9 *
                       optimRelativeFactor) && (MeritFunction->nlpComplError <=
         1.0E-9 * optimRelativeFactor)) {
      Flags->done = true;
      TrialState->sqpExitFlag = 1;
    } else {
      Flags->done = false;
      if (isFeasible && (TrialState->sqpFval < -1.0E+20)) {
        Flags->done = true;
        TrialState->sqpExitFlag = -3;
      } else {
        bool guard1 = false;
        guard1 = false;
        if (TrialState->sqpIterations > 0) {
          dxTooSmall = true;
          idx_max = 0;
          exitg1 = false;
          while ((!exitg1) && (idx_max <= nVar - 1)) {
            if (1.0E-9 * fmax(1.0, fabs(TrialState->xstarsqp[idx_max])) <= fabs
                (TrialState->delta_x[idx_max])) {
              dxTooSmall = false;
              exitg1 = true;
            } else {
              idx_max++;
            }
          }

          if (dxTooSmall) {
            if (!isFeasible) {
              if (Flags->stepType != 2) {
                Flags->stepType = 2;
                Flags->failedLineSearch = false;
                Flags->stepAccepted = false;
                guard1 = true;
              } else {
                Flags->done = true;
                TrialState->sqpExitFlag = -2;
              }
            } else {
              idx_max = WorkingSet->nActiveConstr - 1;
              if (WorkingSet->nActiveConstr > 0) {
                for (b_k = 0; b_k <= idx_max; b_k++) {
                  TrialState->lambda[b_k] = 0.0;
                  idxFiniteLB = 13 * b_k;
                  mLB = 25 * b_k;
                  for (mUB = 0; mUB < nVar; mUB++) {
                    QRManager->QR[mLB + mUB] = WorkingSet->ATwset[idxFiniteLB +
                      mUB];
                  }
                }

                QRManager->usedPivoting = true;
                QRManager->mrows = WorkingSet->nVar;
                QRManager->ncols = WorkingSet->nActiveConstr;
                idx_max = WorkingSet->nVar;
                mUB = WorkingSet->nActiveConstr;
                if (idx_max <= mUB) {
                  mUB = idx_max;
                }

                QRManager->minRowCol = mUB;
                xgeqp3(QRManager->QR, WorkingSet->nVar,
                       WorkingSet->nActiveConstr, QRManager->jpvt,
                       QRManager->tau);
                computeQ_(QRManager, WorkingSet->nVar);
                idx_max = WorkingSet->nVar;
                idxFiniteLB = WorkingSet->nActiveConstr;
                if (idx_max >= idxFiniteLB) {
                  idxFiniteLB = idx_max;
                }

                smax = fabs(QRManager->QR[0]) * fmin(1.4901161193847656E-8,
                  (double)idxFiniteLB * 2.2204460492503131E-16);
                mLB = 0;
                idx_max = 0;
                while ((mLB < mUB) && (fabs(QRManager->QR[idx_max]) > smax)) {
                  mLB++;
                  idx_max += 26;
                }

                xgemv(WorkingSet->nVar, WorkingSet->nVar, QRManager->Q,
                      TrialState->grad, memspace->workspace_double);
                if (mLB != 0) {
                  for (nVar = mLB; nVar >= 1; nVar--) {
                    idx_max = (nVar + (nVar - 1) * 25) - 1;
                    memspace->workspace_double[nVar - 1] /= QRManager->
                      QR[idx_max];
                    for (b_k = 0; b_k <= nVar - 2; b_k++) {
                      idxFiniteLB = (nVar - b_k) - 2;
                      memspace->workspace_double[idxFiniteLB] -=
                        memspace->workspace_double[nVar - 1] * QRManager->QR
                        [(idx_max - b_k) - 1];
                    }
                  }
                }

                idx_max = WorkingSet->nActiveConstr;
                if (idx_max <= mUB) {
                  mUB = idx_max;
                }

                for (idx_max = 0; idx_max < mUB; idx_max++) {
                  TrialState->lambda[QRManager->jpvt[idx_max] - 1] =
                    memspace->workspace_double[idx_max];
                }

                idx_max = WorkingSet->sizes[0] + 1;
                if ((mFixed - idx_max) + 1 < 400) {
                  for (k = idx_max; k <= mFixed; k++) {
                    TrialState->lambda[k - 1] = -TrialState->lambda[k - 1];
                  }
                } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

                  for (k = idx_max; k <= mFixed; k++) {
                    TrialState->lambda[k - 1] = -TrialState->lambda[k - 1];
                  }
                }

                sortLambdaQP(TrialState->lambda, WorkingSet->nActiveConstr,
                             WorkingSet->sizes, WorkingSet->isActiveIdx,
                             WorkingSet->Wid, WorkingSet->Wlocalidx,
                             memspace->workspace_double);
                b_computeGradLag(memspace->workspace_double, WorkingSet->nVar,
                                 TrialState->grad, WorkingSet->indexFixed,
                                 WorkingSet->sizes[0], WorkingSet->indexLB,
                                 WorkingSet->sizes[3], WorkingSet->indexUB,
                                 WorkingSet->sizes[4], TrialState->lambda);
                smax = 0.0;
                idx_max = 0;
                while ((idx_max <= WorkingSet->nVar - 1) && ((!rtIsInf
                         (memspace->workspace_double[idx_max])) && (!rtIsNaN
                         (memspace->workspace_double[idx_max])))) {
                  smax = fmax(smax, fabs(memspace->workspace_double[idx_max]));
                  idx_max++;
                }

                s = computeComplError(TrialState->xstarsqp, WorkingSet->indexLB,
                                      WorkingSet->sizes[3], lb,
                                      WorkingSet->indexUB, WorkingSet->sizes[4],
                                      ub, TrialState->lambda, WorkingSet->sizes
                                      [0] + 1);
                if ((smax <= 1.0E-9 * optimRelativeFactor) && (s <= 1.0E-9 *
                     optimRelativeFactor)) {
                  MeritFunction->nlpDualFeasError = smax;
                  MeritFunction->nlpComplError = s;
                  MeritFunction->firstOrderOpt = fmax(smax, s);
                  if (mLambda + 1 < 400) {
                    if (mLambda >= 0) {
                      memcpy(&TrialState->lambdaStopTest[0], &TrialState->
                             lambda[0], (mLambda + 1) * sizeof(double));
                    }
                  } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

                    for (k = 0; k <= mLambda; k++) {
                      TrialState->lambdaStopTest[k] = TrialState->lambda[k];
                    }
                  }

                  Flags->done = true;
                  TrialState->sqpExitFlag = 1;
                } else {
                  Flags->done = true;
                  TrialState->sqpExitFlag = 2;
                }
              } else {
                Flags->done = true;
                TrialState->sqpExitFlag = 2;
              }
            }
          } else {
            guard1 = true;
          }
        } else {
          guard1 = true;
        }

        if (guard1) {
          if (TrialState->sqpIterations >= 60) {
            Flags->done = true;
            TrialState->sqpExitFlag = 0;
          } else if (TrialState->FunctionEvaluations >= 120) {
            Flags->done = true;
            TrialState->sqpExitFlag = 0;
          }
        }
      }
    }
  }
}

/*
 * Arguments    : double *outTime_tv_sec
 *                double *outTime_tv_nsec
 * Return Type  : void
 */
static void b_timeKeeper(double *outTime_tv_sec, double *outTime_tv_nsec)
{
  *outTime_tv_sec = savedTime.tv_sec;
  *outTime_tv_nsec = savedTime.tv_nsec;
}

/*
 * Arguments    : int n
 *                double a
 *                const double x[72]
 *                int ix0
 *                double y[12]
 *                int iy0
 * Return Type  : void
 */
static void b_xaxpy(int n, double a, const double x[72], int ix0, double y[12],
                    int iy0)
{
  int i1;
  int k;
  if (!(a == 0.0)) {
    int i;
    int ix;
    int iy;
    ix = ix0 - 1;
    iy = iy0 - 1;
    i = n - 1;
    if (n < 400) {
      for (k = 0; k <= i; k++) {
        ix = (iy0 + k) - 1;
        y[ix] += a * x[(ix0 + k) - 1];
      }
    } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4) \
 private(i1)

      for (k = 0; k <= i; k++) {
        i1 = iy + k;
        y[i1] += a * x[ix + k];
      }
    }
  }
}

/*
 * Arguments    : int n
 *                const double x[36]
 *                int ix0
 *                const double y[36]
 *                int iy0
 * Return Type  : double
 */
static double b_xdotc(int n, const double x[36], int ix0, const double y[36],
                      int iy0)
{
  double d;
  int k;
  d = 0.0;
  for (k = 0; k < n; k++) {
    d += x[(ix0 + k) - 1] * y[(iy0 + k) - 1];
  }

  return d;
}

/*
 * Arguments    : int m
 *                int n
 *                int k
 *                const double A[625]
 *                int ia0
 *                const double B[325]
 *                double C[625]
 * Return Type  : void
 */
static void b_xgemm(int m, int n, int k, const double A[625], int ia0, const
                    double B[325], double C[625])
{
  int cr;
  int ic;
  int w;
  if ((m != 0) && (n != 0)) {
    int br;
    int i;
    int i1;
    int lastColC;
    lastColC = 25 * (n - 1);
    for (cr = 0; cr <= lastColC; cr += 25) {
      i = cr + 1;
      i1 = cr + m;
      if (i <= i1) {
        memset(&C[i + -1], 0, ((i1 - i) + 1) * sizeof(double));
      }
    }

    br = -1;
    for (cr = 0; cr <= lastColC; cr += 25) {
      int ar;
      ar = ia0;
      i = cr + 1;
      i1 = cr + m;
      for (ic = i; ic <= i1; ic++) {
        double temp;
        temp = 0.0;
        for (w = 0; w < k; w++) {
          temp += A[(w + ar) - 1] * B[(w + br) + 1];
        }

        C[ic - 1] += temp;
        ar += 25;
      }

      br += 25;
    }
  }
}

/*
 * Arguments    : int n
 *                const double x[13]
 * Return Type  : double
 */
static double b_xnrm2(int n, const double x[13])
{
  double y;
  int k;
  y = 0.0;
  if (n >= 1) {
    if (n == 1) {
      y = fabs(x[0]);
    } else {
      double scale;
      scale = 3.3121686421112381E-170;
      for (k = 0; k < n; k++) {
        double absxk;
        absxk = fabs(x[k]);
        if (absxk > scale) {
          double t;
          t = scale / absxk;
          y = y * t * t + 1.0;
          scale = absxk;
        } else {
          double t;
          t = absxk / scale;
          y += t * t;
        }
      }

      y = scale * sqrt(y);
    }
  }

  return y;
}

/*
 * Arguments    : double x[72]
 *                int ix0
 *                int iy0
 *                double c
 *                double s
 * Return Type  : void
 */
static void b_xrot(double x[72], int ix0, int iy0, double c, double s)
{
  int k;
  for (k = 0; k < 12; k++) {
    double b_temp_tmp;
    double d_temp_tmp;
    int c_temp_tmp;
    int temp_tmp;
    temp_tmp = (iy0 + k) - 1;
    b_temp_tmp = x[temp_tmp];
    c_temp_tmp = (ix0 + k) - 1;
    d_temp_tmp = x[c_temp_tmp];
    x[temp_tmp] = c * b_temp_tmp - s * d_temp_tmp;
    x[c_temp_tmp] = c * d_temp_tmp + s * b_temp_tmp;
  }
}

/*
 * Arguments    : double x[72]
 *                int ix0
 *                int iy0
 * Return Type  : void
 */
static void b_xswap(double x[72], int ix0, int iy0)
{
  int k;
  for (k = 0; k < 12; k++) {
    double temp;
    int i;
    int temp_tmp;
    temp_tmp = (ix0 + k) - 1;
    temp = x[temp_tmp];
    i = (iy0 + k) - 1;
    x[temp_tmp] = x[i];
    x[i] = temp;
  }
}

/*
 * Testing parameters:
 *  (9e-06, 1.31e-07, 2.35, 0.15, 0.13, 0.2, 0.231, 0.231, 0.39, 0.39, 0, ...
 *  0, 0, 0, 100, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0,...
 *  1, 1, 1, 0,...
 *  1, 100, 400, -50, 500, -60,...
 *  0.01, 0.01, 0.01, 0.1, 0.1, 0.01,...
 *  900, 0, 25, -130, 45, -45, 100, -30, 40, [0 0 -5 0 0 0]', 0, 0, 0, 0.1, 5.18, 0.38, 0.2, ...
 *  -.1, 1.225, 0, 0.55, 0.33, 0, 15, -15, 0, 1e-07,...
 *  0, 0, 0, 0, 0)
 *
 * Arguments    : double K_p_T
 *                double K_p_M
 *                double m
 *                double I_xx
 *                double I_yy
 *                double I_zz
 *                double l_1
 *                double l_2
 *                double l_3
 *                double l_4
 *                double l_z
 *                double Phi
 *                double Theta
 *                double Psi
 *                double Omega_1
 *                double Omega_2
 *                double Omega_3
 *                double Omega_4
 *                double b_1
 *                double b_2
 *                double b_3
 *                double b_4
 *                double g_1
 *                double g_2
 *                double g_3
 *                double g_4
 *                double max_omega
 *                double min_omega
 *                double max_b
 *                double min_b
 *                double max_g
 *                double min_g
 *                double dv[6]
 *                double p
 *                double q
 *                double r
 *                double Cm_zero
 *                double Cl_alpha
 *                double Cd_zero
 *                double K_Cd
 *                double Cm_alpha
 *                double rho
 *                double V
 *                double S
 *                double wing_chord
 *                double flight_path_angle
 *                double Beta
 *                double desired_motor_value
 *                double desired_el_value
 *                double desired_az_value
 *                double u_out[12]
 *                double residuals[6]
 *                double *elapsed_time
 *                double *exitflag
 * Return Type  : void
 */
static void c_Basic_inversion_controller_fc(double K_p_T, double K_p_M, double m,
  double I_xx, double I_yy, double I_zz, double l_1, double l_2, double l_3,
  double l_4, double l_z, double Phi, double Theta, double Psi, double Omega_1,
  double Omega_2, double Omega_3, double Omega_4, double b_1, double b_2, double
  b_3, double b_4, double g_1, double g_2, double g_3, double g_4, double
  max_omega, double min_omega, double max_b, double min_b, double max_g, double
  min_g, double dv[6], double p, double q, double r, double Cm_zero, double
  Cl_alpha, double Cd_zero, double K_Cd, double Cm_alpha, double rho, double V,
  double S, double wing_chord, double flight_path_angle, double Beta, double
  desired_motor_value, double desired_el_value, double desired_az_value, double
  u_out[12], double residuals[6], double *elapsed_time, double *exitflag)
{
  double A[72];
  double B_eval[72];
  double actual_u[12];
  double u_out_feasible[12];
  double varargin_2[12];
  double x[12];
  double s[6];
  double B_eval_tmp;
  double B_eval_tmp_tmp;
  double ab_B_eval_tmp;
  double absx;
  double ac_B_eval_tmp;
  double b_B_eval_tmp;
  double b_B_eval_tmp_tmp;
  double bb_B_eval_tmp;
  double c_B_eval_tmp;
  double c_B_eval_tmp_tmp;
  double cb_B_eval_tmp;
  double d_B_eval_tmp;
  double db_B_eval_tmp;
  double e_B_eval_tmp;
  double eb_B_eval_tmp;
  double f_B_eval_tmp;
  double fb_B_eval_tmp;
  double g_B_eval_tmp;
  double gain_az;
  double gain_el;
  double gain_motor;
  double gb_B_eval_tmp;
  double h_B_eval_tmp;
  double hb_B_eval_tmp;
  double i_B_eval_tmp;
  double ib_B_eval_tmp;
  double j_B_eval_tmp;
  double jb_B_eval_tmp;
  double k_B_eval_tmp;
  double kb_B_eval_tmp;
  double l_B_eval_tmp;
  double lb_B_eval_tmp;
  double m_B_eval_tmp;
  double mb_B_eval_tmp;
  double n_B_eval_tmp;
  double nb_B_eval_tmp;
  double o_B_eval_tmp;
  double ob_B_eval_tmp;
  double p_B_eval_tmp;
  double pb_B_eval_tmp;
  double q_B_eval_tmp;
  double qb_B_eval_tmp;
  double r_B_eval_tmp;
  double rb_B_eval_tmp;
  double s_B_eval_tmp;
  double sb_B_eval_tmp;
  double t_B_eval_tmp;
  double tb_B_eval_tmp;
  double u_B_eval_tmp;
  double ub_B_eval_tmp;
  double v_B_eval_tmp;
  double vb_B_eval_tmp;
  double w_B_eval_tmp;
  double wb_B_eval_tmp;
  double x_B_eval_tmp;
  double xb_B_eval_tmp;
  double y_B_eval_tmp;
  double yb_B_eval_tmp;
  int ar;
  int exponent;
  int i;
  int i1;
  int ib;
  int ic;
  int vcol;
  bool b_x[12];
  bool b_p;
  tic();

  /*  Assign geometrical and create variables */
  gain_motor = max_omega / 2.0;
  gain_el = (max_b - min_b) * 3.1415926535897931 / 180.0 / 2.0;
  gain_az = (max_g - min_g) * 3.1415926535897931 / 180.0 / 2.0;

  /*  Identify and evaluate the effectiveness matrix [6,12] */
  B_eval_tmp = cos(Psi);
  b_B_eval_tmp = cos(Phi);
  c_B_eval_tmp = sin(Psi);
  d_B_eval_tmp = sin(Phi);
  e_B_eval_tmp = sin(Theta);
  f_B_eval_tmp = cos(Theta);
  g_B_eval_tmp = cos(g_1);
  h_B_eval_tmp = sin(b_1);
  i_B_eval_tmp = cos(b_1);
  j_B_eval_tmp = sin(g_1);
  k_B_eval_tmp = cos(g_2);
  l_B_eval_tmp = sin(b_2);
  m_B_eval_tmp = cos(b_2);
  n_B_eval_tmp = sin(g_2);
  o_B_eval_tmp = cos(g_3);
  p_B_eval_tmp = sin(b_3);
  q_B_eval_tmp = cos(b_3);
  r_B_eval_tmp = sin(g_3);
  s_B_eval_tmp = cos(g_4);
  t_B_eval_tmp = sin(b_4);
  u_B_eval_tmp = cos(b_4);
  v_B_eval_tmp = sin(g_4);
  w_B_eval_tmp = 2.0 * K_p_T * Omega_1;
  B_eval[0] = -((w_B_eval_tmp * B_eval_tmp * f_B_eval_tmp * h_B_eval_tmp +
                 w_B_eval_tmp * i_B_eval_tmp * g_B_eval_tmp * (d_B_eval_tmp *
    c_B_eval_tmp + b_B_eval_tmp * B_eval_tmp * e_B_eval_tmp)) + 2.0 * K_p_T
                * Omega_1 * cos(b_1) * j_B_eval_tmp * (b_B_eval_tmp *
    c_B_eval_tmp - B_eval_tmp * d_B_eval_tmp * e_B_eval_tmp)) / m;
  x_B_eval_tmp = 2.0 * K_p_T * Omega_2;
  y_B_eval_tmp = sin(Phi) * sin(Psi) + cos(Phi) * cos(Psi) * sin(Theta);
  ab_B_eval_tmp = cos(Phi) * sin(Psi) - cos(Psi) * sin(Phi) * sin(Theta);
  B_eval[6] = -((x_B_eval_tmp * B_eval_tmp * f_B_eval_tmp * l_B_eval_tmp +
                 x_B_eval_tmp * m_B_eval_tmp * k_B_eval_tmp * y_B_eval_tmp) +
                2.0 * K_p_T * Omega_2 * cos(b_2) * n_B_eval_tmp * ab_B_eval_tmp)
    / m;
  bb_B_eval_tmp = 2.0 * K_p_T * Omega_3;
  B_eval[12] = -((bb_B_eval_tmp * B_eval_tmp * f_B_eval_tmp * p_B_eval_tmp +
                  bb_B_eval_tmp * q_B_eval_tmp * o_B_eval_tmp * y_B_eval_tmp) +
                 2.0 * K_p_T * Omega_3 * cos(b_3) * r_B_eval_tmp * ab_B_eval_tmp)
    / m;
  cb_B_eval_tmp = 2.0 * K_p_T * Omega_4;
  B_eval[18] = -((cb_B_eval_tmp * B_eval_tmp * f_B_eval_tmp * t_B_eval_tmp +
                  cb_B_eval_tmp * u_B_eval_tmp * s_B_eval_tmp * y_B_eval_tmp) +
                 2.0 * K_p_T * Omega_4 * cos(b_4) * v_B_eval_tmp * ab_B_eval_tmp)
    / m;
  absx = Omega_1 * Omega_1;
  db_B_eval_tmp = K_p_T * absx;
  eb_B_eval_tmp = db_B_eval_tmp * g_B_eval_tmp * h_B_eval_tmp;
  fb_B_eval_tmp = db_B_eval_tmp * h_B_eval_tmp * j_B_eval_tmp;
  B_eval[24] = ((eb_B_eval_tmp * y_B_eval_tmp - db_B_eval_tmp * B_eval_tmp *
                 f_B_eval_tmp * i_B_eval_tmp) + fb_B_eval_tmp * ab_B_eval_tmp) /
    m;
  B_eval_tmp_tmp = Omega_2 * Omega_2;
  gb_B_eval_tmp = K_p_T * B_eval_tmp_tmp;
  hb_B_eval_tmp = gb_B_eval_tmp * k_B_eval_tmp * l_B_eval_tmp;
  ib_B_eval_tmp = gb_B_eval_tmp * l_B_eval_tmp * n_B_eval_tmp;
  B_eval[30] = ((hb_B_eval_tmp * y_B_eval_tmp - gb_B_eval_tmp * B_eval_tmp *
                 f_B_eval_tmp * m_B_eval_tmp) + ib_B_eval_tmp * ab_B_eval_tmp) /
    m;
  b_B_eval_tmp_tmp = Omega_3 * Omega_3;
  jb_B_eval_tmp = K_p_T * b_B_eval_tmp_tmp;
  kb_B_eval_tmp = jb_B_eval_tmp * o_B_eval_tmp * p_B_eval_tmp;
  lb_B_eval_tmp = jb_B_eval_tmp * p_B_eval_tmp * r_B_eval_tmp;
  B_eval[36] = ((kb_B_eval_tmp * y_B_eval_tmp - jb_B_eval_tmp * B_eval_tmp *
                 f_B_eval_tmp * q_B_eval_tmp) + lb_B_eval_tmp * ab_B_eval_tmp) /
    m;
  c_B_eval_tmp_tmp = Omega_4 * Omega_4;
  mb_B_eval_tmp = K_p_T * c_B_eval_tmp_tmp;
  nb_B_eval_tmp = mb_B_eval_tmp * s_B_eval_tmp * t_B_eval_tmp;
  ob_B_eval_tmp = mb_B_eval_tmp * t_B_eval_tmp * v_B_eval_tmp;
  B_eval[42] = ((nb_B_eval_tmp * y_B_eval_tmp - mb_B_eval_tmp * B_eval_tmp *
                 f_B_eval_tmp * u_B_eval_tmp) + ob_B_eval_tmp * ab_B_eval_tmp) /
    m;
  B_eval_tmp = db_B_eval_tmp * i_B_eval_tmp;
  pb_B_eval_tmp = B_eval_tmp * g_B_eval_tmp;
  qb_B_eval_tmp = B_eval_tmp * j_B_eval_tmp;
  B_eval[48] = -(pb_B_eval_tmp * ab_B_eval_tmp - qb_B_eval_tmp * y_B_eval_tmp) /
    m;
  rb_B_eval_tmp = gb_B_eval_tmp * m_B_eval_tmp;
  sb_B_eval_tmp = rb_B_eval_tmp * k_B_eval_tmp;
  tb_B_eval_tmp = rb_B_eval_tmp * n_B_eval_tmp;
  B_eval[54] = -(sb_B_eval_tmp * ab_B_eval_tmp - tb_B_eval_tmp * y_B_eval_tmp) /
    m;
  ub_B_eval_tmp = jb_B_eval_tmp * q_B_eval_tmp;
  vb_B_eval_tmp = ub_B_eval_tmp * o_B_eval_tmp;
  wb_B_eval_tmp = ub_B_eval_tmp * r_B_eval_tmp;
  B_eval[60] = -(vb_B_eval_tmp * ab_B_eval_tmp - wb_B_eval_tmp * y_B_eval_tmp) /
    m;
  xb_B_eval_tmp = mb_B_eval_tmp * u_B_eval_tmp;
  yb_B_eval_tmp = xb_B_eval_tmp * s_B_eval_tmp;
  ac_B_eval_tmp = xb_B_eval_tmp * v_B_eval_tmp;
  B_eval[66] = -(yb_B_eval_tmp * ab_B_eval_tmp - ac_B_eval_tmp * y_B_eval_tmp) /
    m;
  B_eval[1] = ((2.0 * K_p_T * Omega_1 * cos(b_1) * cos(g_1) * (cos(Psi) * sin
    (Phi) - cos(Phi) * sin(Psi) * e_B_eval_tmp) - w_B_eval_tmp * f_B_eval_tmp *
                c_B_eval_tmp * h_B_eval_tmp) + 2.0 * K_p_T * Omega_1 * cos(b_1) *
               sin(g_1) * (cos(Phi) * cos(Psi) + sin(Phi) * sin(Psi) *
    e_B_eval_tmp)) / m;
  e_B_eval_tmp = cos(Psi) * sin(Phi) - cos(Phi) * sin(Psi) * sin(Theta);
  y_B_eval_tmp = cos(Phi) * cos(Psi) + sin(Phi) * sin(Psi) * sin(Theta);
  B_eval[7] = ((2.0 * K_p_T * Omega_2 * cos(b_2) * cos(g_2) * e_B_eval_tmp -
                x_B_eval_tmp * f_B_eval_tmp * c_B_eval_tmp * l_B_eval_tmp) + 2.0
               * K_p_T * Omega_2 * cos(b_2) * sin(g_2) * y_B_eval_tmp) / m;
  B_eval[13] = ((2.0 * K_p_T * Omega_3 * cos(b_3) * cos(g_3) * e_B_eval_tmp -
                 bb_B_eval_tmp * f_B_eval_tmp * c_B_eval_tmp * p_B_eval_tmp) +
                2.0 * K_p_T * Omega_3 * cos(b_3) * sin(g_3) * y_B_eval_tmp) / m;
  B_eval[19] = ((2.0 * K_p_T * Omega_4 * cos(b_4) * cos(g_4) * e_B_eval_tmp -
                 cb_B_eval_tmp * f_B_eval_tmp * c_B_eval_tmp * t_B_eval_tmp) +
                2.0 * K_p_T * Omega_4 * cos(b_4) * sin(g_4) * y_B_eval_tmp) / m;
  ab_B_eval_tmp = db_B_eval_tmp * f_B_eval_tmp;
  B_eval[25] = -((ab_B_eval_tmp * c_B_eval_tmp * i_B_eval_tmp + eb_B_eval_tmp *
                  e_B_eval_tmp) + fb_B_eval_tmp * y_B_eval_tmp) / m;
  eb_B_eval_tmp = gb_B_eval_tmp * f_B_eval_tmp;
  B_eval[31] = -((eb_B_eval_tmp * c_B_eval_tmp * m_B_eval_tmp + hb_B_eval_tmp *
                  e_B_eval_tmp) + ib_B_eval_tmp * y_B_eval_tmp) / m;
  fb_B_eval_tmp = jb_B_eval_tmp * f_B_eval_tmp;
  B_eval[37] = -((fb_B_eval_tmp * c_B_eval_tmp * q_B_eval_tmp + kb_B_eval_tmp *
                  e_B_eval_tmp) + lb_B_eval_tmp * y_B_eval_tmp) / m;
  hb_B_eval_tmp = mb_B_eval_tmp * f_B_eval_tmp;
  B_eval[43] = -((hb_B_eval_tmp * c_B_eval_tmp * u_B_eval_tmp + nb_B_eval_tmp *
                  e_B_eval_tmp) + ob_B_eval_tmp * y_B_eval_tmp) / m;
  B_eval[49] = (pb_B_eval_tmp * y_B_eval_tmp - qb_B_eval_tmp * e_B_eval_tmp) / m;
  B_eval[55] = (sb_B_eval_tmp * y_B_eval_tmp - tb_B_eval_tmp * e_B_eval_tmp) / m;
  B_eval[61] = (vb_B_eval_tmp * y_B_eval_tmp - wb_B_eval_tmp * e_B_eval_tmp) / m;
  B_eval[67] = (yb_B_eval_tmp * y_B_eval_tmp - ac_B_eval_tmp * e_B_eval_tmp) / m;
  B_eval[2] = ((w_B_eval_tmp * h_B_eval_tmp * e_B_eval_tmp + 2.0 * K_p_T
                * Omega_1 * cos(Theta) * d_B_eval_tmp * i_B_eval_tmp *
                j_B_eval_tmp) - w_B_eval_tmp * b_B_eval_tmp * f_B_eval_tmp *
               i_B_eval_tmp * g_B_eval_tmp) / m;
  B_eval[8] = ((x_B_eval_tmp * l_B_eval_tmp * e_B_eval_tmp + 2.0 * K_p_T
                * Omega_2 * cos(Theta) * d_B_eval_tmp * m_B_eval_tmp *
                n_B_eval_tmp) - x_B_eval_tmp * b_B_eval_tmp * f_B_eval_tmp *
               m_B_eval_tmp * k_B_eval_tmp) / m;
  B_eval[14] = ((bb_B_eval_tmp * p_B_eval_tmp * e_B_eval_tmp + 2.0 * K_p_T
                 * Omega_3 * cos(Theta) * d_B_eval_tmp * q_B_eval_tmp *
                 r_B_eval_tmp) - bb_B_eval_tmp * b_B_eval_tmp * f_B_eval_tmp *
                q_B_eval_tmp * o_B_eval_tmp) / m;
  B_eval[20] = ((cb_B_eval_tmp * t_B_eval_tmp * e_B_eval_tmp + 2.0 * K_p_T
                 * Omega_4 * cos(Theta) * d_B_eval_tmp * u_B_eval_tmp *
                 v_B_eval_tmp) - cb_B_eval_tmp * b_B_eval_tmp * f_B_eval_tmp *
                u_B_eval_tmp * s_B_eval_tmp) / m;
  c_B_eval_tmp = db_B_eval_tmp * b_B_eval_tmp * f_B_eval_tmp;
  y_B_eval_tmp = ab_B_eval_tmp * d_B_eval_tmp;
  B_eval[26] = ((B_eval_tmp * e_B_eval_tmp + c_B_eval_tmp * g_B_eval_tmp *
                 h_B_eval_tmp) - y_B_eval_tmp * h_B_eval_tmp * j_B_eval_tmp) / m;
  B_eval_tmp = gb_B_eval_tmp * b_B_eval_tmp * f_B_eval_tmp;
  ab_B_eval_tmp = eb_B_eval_tmp * d_B_eval_tmp;
  B_eval[32] = ((rb_B_eval_tmp * e_B_eval_tmp + B_eval_tmp * k_B_eval_tmp *
                 l_B_eval_tmp) - ab_B_eval_tmp * l_B_eval_tmp * n_B_eval_tmp) /
    m;
  eb_B_eval_tmp = jb_B_eval_tmp * b_B_eval_tmp * f_B_eval_tmp;
  fb_B_eval_tmp *= d_B_eval_tmp;
  B_eval[38] = ((ub_B_eval_tmp * e_B_eval_tmp + eb_B_eval_tmp * o_B_eval_tmp *
                 p_B_eval_tmp) - fb_B_eval_tmp * p_B_eval_tmp * r_B_eval_tmp) /
    m;
  b_B_eval_tmp = mb_B_eval_tmp * b_B_eval_tmp * f_B_eval_tmp;
  d_B_eval_tmp *= hb_B_eval_tmp;
  B_eval[44] = ((xb_B_eval_tmp * e_B_eval_tmp + b_B_eval_tmp * s_B_eval_tmp *
                 t_B_eval_tmp) - d_B_eval_tmp * t_B_eval_tmp * v_B_eval_tmp) / m;
  B_eval[50] = (c_B_eval_tmp * i_B_eval_tmp * j_B_eval_tmp + y_B_eval_tmp *
                i_B_eval_tmp * g_B_eval_tmp) / m;
  B_eval[56] = (B_eval_tmp * m_B_eval_tmp * n_B_eval_tmp + ab_B_eval_tmp *
                m_B_eval_tmp * k_B_eval_tmp) / m;
  B_eval[62] = (eb_B_eval_tmp * q_B_eval_tmp * r_B_eval_tmp + fb_B_eval_tmp *
                q_B_eval_tmp * o_B_eval_tmp) / m;
  B_eval[68] = (b_B_eval_tmp * u_B_eval_tmp * v_B_eval_tmp + d_B_eval_tmp *
                u_B_eval_tmp * s_B_eval_tmp) / m;
  B_eval_tmp = w_B_eval_tmp * l_z;
  b_B_eval_tmp = 2.0 * K_p_M * Omega_1;
  c_B_eval_tmp = w_B_eval_tmp * l_1;
  B_eval[3] = ((b_B_eval_tmp * h_B_eval_tmp + B_eval_tmp * i_B_eval_tmp *
                j_B_eval_tmp) + c_B_eval_tmp * i_B_eval_tmp * g_B_eval_tmp) /
    I_xx;
  d_B_eval_tmp = x_B_eval_tmp * l_z;
  e_B_eval_tmp = 2.0 * K_p_M * Omega_2;
  f_B_eval_tmp = x_B_eval_tmp * l_1;
  B_eval[9] = -((e_B_eval_tmp * l_B_eval_tmp - d_B_eval_tmp * m_B_eval_tmp *
                 n_B_eval_tmp) + f_B_eval_tmp * m_B_eval_tmp * k_B_eval_tmp) /
    I_xx;
  y_B_eval_tmp = 2.0 * K_p_M * Omega_3;
  ab_B_eval_tmp = bb_B_eval_tmp * l_z;
  eb_B_eval_tmp = bb_B_eval_tmp * l_2;
  B_eval[15] = ((y_B_eval_tmp * p_B_eval_tmp + ab_B_eval_tmp * q_B_eval_tmp *
                 r_B_eval_tmp) - eb_B_eval_tmp * q_B_eval_tmp * o_B_eval_tmp) /
    I_xx;
  fb_B_eval_tmp = cb_B_eval_tmp * l_z;
  hb_B_eval_tmp = 2.0 * K_p_M * Omega_4;
  ib_B_eval_tmp = cb_B_eval_tmp * l_2;
  B_eval[21] = ((fb_B_eval_tmp * u_B_eval_tmp * v_B_eval_tmp - hb_B_eval_tmp *
                 t_B_eval_tmp) + ib_B_eval_tmp * u_B_eval_tmp * s_B_eval_tmp) /
    I_xx;
  kb_B_eval_tmp = db_B_eval_tmp * l_z;
  lb_B_eval_tmp = db_B_eval_tmp * l_1;
  nb_B_eval_tmp = K_p_M * absx;
  ob_B_eval_tmp = nb_B_eval_tmp * i_B_eval_tmp;
  B_eval[27] = -((lb_B_eval_tmp * g_B_eval_tmp * h_B_eval_tmp - ob_B_eval_tmp) +
                 kb_B_eval_tmp * h_B_eval_tmp * j_B_eval_tmp) / I_xx;
  pb_B_eval_tmp = gb_B_eval_tmp * l_z;
  qb_B_eval_tmp = gb_B_eval_tmp * l_1;
  rb_B_eval_tmp = K_p_M * B_eval_tmp_tmp;
  sb_B_eval_tmp = rb_B_eval_tmp * m_B_eval_tmp;
  B_eval[33] = -((sb_B_eval_tmp - qb_B_eval_tmp * k_B_eval_tmp * l_B_eval_tmp) +
                 pb_B_eval_tmp * l_B_eval_tmp * n_B_eval_tmp) / I_xx;
  tb_B_eval_tmp = jb_B_eval_tmp * l_z;
  ub_B_eval_tmp = jb_B_eval_tmp * l_2;
  vb_B_eval_tmp = K_p_M * b_B_eval_tmp_tmp;
  wb_B_eval_tmp = vb_B_eval_tmp * q_B_eval_tmp;
  B_eval[39] = ((wb_B_eval_tmp + ub_B_eval_tmp * o_B_eval_tmp * p_B_eval_tmp) -
                tb_B_eval_tmp * p_B_eval_tmp * r_B_eval_tmp) / I_xx;
  xb_B_eval_tmp = mb_B_eval_tmp * l_z;
  yb_B_eval_tmp = mb_B_eval_tmp * l_2;
  ac_B_eval_tmp = K_p_M * c_B_eval_tmp_tmp;
  absx = ac_B_eval_tmp * u_B_eval_tmp;
  B_eval[45] = -((absx + yb_B_eval_tmp * s_B_eval_tmp * t_B_eval_tmp) +
                 xb_B_eval_tmp * t_B_eval_tmp * v_B_eval_tmp) / I_xx;
  kb_B_eval_tmp *= i_B_eval_tmp;
  lb_B_eval_tmp *= i_B_eval_tmp;
  B_eval[51] = (kb_B_eval_tmp * g_B_eval_tmp - lb_B_eval_tmp * j_B_eval_tmp) /
    I_xx;
  pb_B_eval_tmp *= m_B_eval_tmp;
  qb_B_eval_tmp *= m_B_eval_tmp;
  B_eval[57] = (pb_B_eval_tmp * k_B_eval_tmp + qb_B_eval_tmp * n_B_eval_tmp) /
    I_xx;
  tb_B_eval_tmp *= q_B_eval_tmp;
  ub_B_eval_tmp *= q_B_eval_tmp;
  B_eval[63] = (tb_B_eval_tmp * o_B_eval_tmp + ub_B_eval_tmp * r_B_eval_tmp) /
    I_xx;
  xb_B_eval_tmp *= u_B_eval_tmp;
  yb_B_eval_tmp *= u_B_eval_tmp;
  B_eval[69] = (xb_B_eval_tmp * s_B_eval_tmp - yb_B_eval_tmp * v_B_eval_tmp) /
    I_xx;
  B_eval[4] = ((B_eval_tmp * h_B_eval_tmp - b_B_eval_tmp * i_B_eval_tmp *
                j_B_eval_tmp) + w_B_eval_tmp * l_4 * i_B_eval_tmp * g_B_eval_tmp)
    / I_yy;
  B_eval[10] = ((d_B_eval_tmp * l_B_eval_tmp + e_B_eval_tmp * m_B_eval_tmp *
                 n_B_eval_tmp) + x_B_eval_tmp * l_4 * m_B_eval_tmp *
                k_B_eval_tmp) / I_yy;
  B_eval[16] = -((y_B_eval_tmp * q_B_eval_tmp * r_B_eval_tmp - ab_B_eval_tmp *
                  p_B_eval_tmp) + bb_B_eval_tmp * l_3 * q_B_eval_tmp *
                 o_B_eval_tmp) / I_yy;
  B_eval[22] = ((fb_B_eval_tmp * t_B_eval_tmp + hb_B_eval_tmp * u_B_eval_tmp *
                 v_B_eval_tmp) - cb_B_eval_tmp * l_3 * u_B_eval_tmp *
                s_B_eval_tmp) / I_yy;
  B_eval_tmp = db_B_eval_tmp * l_4;
  B_eval[28] = ((nb_B_eval_tmp * h_B_eval_tmp * j_B_eval_tmp + kb_B_eval_tmp) -
                B_eval_tmp * g_B_eval_tmp * h_B_eval_tmp) / I_yy;
  b_B_eval_tmp = gb_B_eval_tmp * l_4;
  B_eval[34] = -((rb_B_eval_tmp * l_B_eval_tmp * n_B_eval_tmp - pb_B_eval_tmp) +
                 b_B_eval_tmp * k_B_eval_tmp * l_B_eval_tmp) / I_yy;
  d_B_eval_tmp = jb_B_eval_tmp * l_3;
  B_eval[40] = ((vb_B_eval_tmp * p_B_eval_tmp * r_B_eval_tmp + tb_B_eval_tmp) +
                d_B_eval_tmp * o_B_eval_tmp * p_B_eval_tmp) / I_yy;
  e_B_eval_tmp = mb_B_eval_tmp * l_3;
  B_eval[46] = ((xb_B_eval_tmp - ac_B_eval_tmp * t_B_eval_tmp * v_B_eval_tmp) +
                e_B_eval_tmp * s_B_eval_tmp * t_B_eval_tmp) / I_yy;
  i_B_eval_tmp *= B_eval_tmp;
  B_eval[52] = -(ob_B_eval_tmp * g_B_eval_tmp + i_B_eval_tmp * j_B_eval_tmp) /
    I_yy;
  m_B_eval_tmp *= b_B_eval_tmp;
  B_eval[58] = (sb_B_eval_tmp * k_B_eval_tmp - m_B_eval_tmp * n_B_eval_tmp) /
    I_yy;
  q_B_eval_tmp *= d_B_eval_tmp;
  B_eval[64] = -(wb_B_eval_tmp * o_B_eval_tmp - q_B_eval_tmp * r_B_eval_tmp) /
    I_yy;
  u_B_eval_tmp *= e_B_eval_tmp;
  B_eval[70] = (absx * s_B_eval_tmp + u_B_eval_tmp * v_B_eval_tmp) / I_yy;
  B_eval[5] = ((2.0 * K_p_M * Omega_1 * cos(b_1) * g_B_eval_tmp - c_B_eval_tmp *
                h_B_eval_tmp) + 2.0 * K_p_T * Omega_1 * l_4 * cos(b_1) *
               j_B_eval_tmp) / I_zz;
  B_eval[11] = ((f_B_eval_tmp * l_B_eval_tmp - 2.0 * K_p_M * Omega_2 * cos(b_2) *
                 k_B_eval_tmp) + 2.0 * K_p_T * Omega_2 * l_4 * cos(b_2) *
                n_B_eval_tmp) / I_zz;
  B_eval[17] = ((eb_B_eval_tmp * p_B_eval_tmp + 2.0 * K_p_M * Omega_3 * cos(b_3)
                 * o_B_eval_tmp) - 2.0 * K_p_T * Omega_3 * l_3 * cos(b_3) *
                r_B_eval_tmp) / I_zz;
  B_eval[23] = -((ib_B_eval_tmp * t_B_eval_tmp + 2.0 * K_p_M * Omega_4 * cos(b_4)
                  * s_B_eval_tmp) + 2.0 * K_p_T * Omega_4 * l_3 * cos(b_4) *
                 v_B_eval_tmp) / I_zz;
  B_eval[29] = -((lb_B_eval_tmp + nb_B_eval_tmp * g_B_eval_tmp * h_B_eval_tmp) +
                 B_eval_tmp * h_B_eval_tmp * j_B_eval_tmp) / I_zz;
  B_eval[35] = ((qb_B_eval_tmp + rb_B_eval_tmp * k_B_eval_tmp * l_B_eval_tmp) -
                b_B_eval_tmp * l_B_eval_tmp * n_B_eval_tmp) / I_zz;
  B_eval[41] = ((ub_B_eval_tmp - vb_B_eval_tmp * o_B_eval_tmp * p_B_eval_tmp) +
                d_B_eval_tmp * p_B_eval_tmp * r_B_eval_tmp) / I_zz;
  B_eval[47] = ((ac_B_eval_tmp * s_B_eval_tmp * t_B_eval_tmp - yb_B_eval_tmp) +
                e_B_eval_tmp * t_B_eval_tmp * v_B_eval_tmp) / I_zz;
  B_eval[53] = -(ob_B_eval_tmp * j_B_eval_tmp - i_B_eval_tmp * g_B_eval_tmp) /
    I_zz;
  B_eval[59] = (sb_B_eval_tmp * n_B_eval_tmp + m_B_eval_tmp * k_B_eval_tmp) /
    I_zz;
  B_eval[65] = -(wb_B_eval_tmp * r_B_eval_tmp + q_B_eval_tmp * o_B_eval_tmp) /
    I_zz;
  B_eval[71] = (absx * v_B_eval_tmp - u_B_eval_tmp * s_B_eval_tmp) / I_zz;
  actual_u[0] = Omega_1;
  actual_u[1] = Omega_2;
  actual_u[2] = Omega_3;
  actual_u[3] = Omega_4;
  actual_u[4] = b_1;
  actual_u[5] = b_2;
  actual_u[6] = b_3;
  actual_u[7] = b_4;
  actual_u[8] = g_1;
  actual_u[9] = g_2;
  actual_u[10] = g_3;
  actual_u[11] = g_4;
  c_compute_acc_nonlinear_earth_r(actual_u, Theta, Phi, Psi, p, q, r, K_p_T,
    K_p_M, m, I_xx, I_yy, I_zz, l_1, l_2, l_3, l_4, l_z, Cl_alpha, Cd_zero, K_Cd,
    Cm_alpha, Cm_zero, rho, V, S, wing_chord, flight_path_angle, Beta, residuals);
  for (i = 0; i < 6; i++) {
    residuals[i] += dv[i];
  }

  u_out[0] = desired_motor_value;
  u_out[1] = desired_motor_value;
  u_out[2] = desired_motor_value;
  u_out[3] = desired_motor_value;
  u_out[4] = desired_el_value;
  u_out[5] = desired_el_value;
  u_out[6] = desired_el_value;
  u_out[7] = desired_el_value;
  u_out[8] = desired_az_value;
  u_out[9] = desired_az_value;
  u_out[10] = desired_az_value;
  u_out[11] = desired_az_value;
  for (i = 0; i < 12; i++) {
    actual_u[i] -= u_out[i];
  }

  for (i = 0; i < 6; i++) {
    absx = 0.0;
    for (i1 = 0; i1 < 12; i1++) {
      absx += B_eval[i + 6 * i1] * actual_u[i1];
    }

    dv[i] += absx;
  }

  /*  Apply gains for the normalization of the effectiveness matrix: */
  for (i = 0; i < 4; i++) {
    for (i1 = 0; i1 < 6; i1++) {
      vcol = i1 + 6 * i;
      B_eval[vcol] *= gain_motor;
      vcol = i1 + 6 * (i + 4);
      B_eval[vcol] *= gain_el;
      vcol = i1 + 6 * (i + 8);
      B_eval[vcol] *= gain_az;
    }
  }

  /*  Apply basic inversion to compute the control input array: */
  for (i = 0; i < 6; i++) {
    for (i1 = 0; i1 < 12; i1++) {
      A[i1 + 12 * i] = B_eval[i + 6 * i1];
    }
  }

  b_p = true;
  for (ar = 0; ar < 72; ar++) {
    B_eval[ar] = 0.0;
    if ((!b_p) || (rtIsInf(A[ar]) || rtIsNaN(A[ar]))) {
      b_p = false;
    }
  }

  if (!b_p) {
    for (i = 0; i < 72; i++) {
      B_eval[i] = rtNaN;
    }
  } else {
    double U[72];
    double b_V[36];
    int b_r;
    svd(A, U, s, b_V);
    absx = fabs(s[0]);
    if ((!rtIsInf(absx)) && (!rtIsNaN(absx))) {
      if (absx <= 2.2250738585072014E-308) {
        absx = 4.94065645841247E-324;
      } else {
        frexp(absx, &exponent);
        absx = ldexp(1.0, exponent - 53);
      }
    } else {
      absx = rtNaN;
    }

    absx *= 12.0;
    b_r = -1;
    ar = 0;
    while ((ar < 6) && (s[ar] > absx)) {
      b_r++;
      ar++;
    }

    if (b_r + 1 > 0) {
      vcol = 1;
      for (exponent = 0; exponent <= b_r; exponent++) {
        absx = 1.0 / s[exponent];
        i = vcol + 5;
        for (ar = vcol; ar <= i; ar++) {
          b_V[ar - 1] *= absx;
        }

        vcol += 6;
      }

      for (exponent = 0; exponent <= 66; exponent += 6) {
        i = exponent + 1;
        i1 = exponent + 6;
        if (i <= i1) {
          memset(&B_eval[i + -1], 0, ((i1 - i) + 1) * sizeof(double));
        }
      }

      vcol = 0;
      for (exponent = 0; exponent <= 66; exponent += 6) {
        ar = -1;
        vcol++;
        i = vcol + 12 * b_r;
        for (ib = vcol; ib <= i; ib += 12) {
          int i2;
          i1 = exponent + 1;
          i2 = exponent + 6;
          for (ic = i1; ic <= i2; ic++) {
            B_eval[ic - 1] += U[ib - 1] * b_V[(ar + ic) - exponent];
          }

          ar += 6;
        }
      }
    }
  }

  for (i = 0; i < 12; i++) {
    absx = 0.0;
    for (i1 = 0; i1 < 6; i1++) {
      absx += B_eval[i1 + 6 * i] * dv[i1];
    }

    actual_u[i] = absx;
  }

  /* Scale back the increment to remove the normalization */
  actual_u[0] *= gain_motor;
  actual_u[4] *= gain_el;
  actual_u[8] *= gain_az;
  actual_u[1] *= gain_motor;
  actual_u[5] *= gain_el;
  actual_u[9] *= gain_az;
  actual_u[2] *= gain_motor;
  actual_u[6] *= gain_el;
  actual_u[10] *= gain_az;
  actual_u[3] *= gain_motor;
  actual_u[7] *= gain_el;
  actual_u[11] *= gain_az;
  varargin_2[0] = max_omega;
  varargin_2[1] = max_omega;
  varargin_2[2] = max_omega;
  varargin_2[3] = max_omega;
  varargin_2[4] = max_b;
  varargin_2[5] = max_b;
  varargin_2[6] = max_b;
  varargin_2[7] = max_b;
  varargin_2[8] = max_g;
  varargin_2[9] = max_g;
  varargin_2[10] = max_g;
  varargin_2[11] = max_g;
  for (ar = 0; ar < 12; ar++) {
    absx = actual_u[ar] + u_out[ar];
    u_out[ar] = absx;
    x[ar] = fmin(absx, varargin_2[ar]);
  }

  varargin_2[0] = min_omega;
  varargin_2[1] = min_omega;
  varargin_2[2] = min_omega;
  varargin_2[3] = min_omega;
  varargin_2[4] = min_b;
  varargin_2[5] = min_b;
  varargin_2[6] = min_b;
  varargin_2[7] = min_b;
  varargin_2[8] = min_g;
  varargin_2[9] = min_g;
  varargin_2[10] = min_g;
  varargin_2[11] = min_g;
  for (ar = 0; ar < 12; ar++) {
    u_out_feasible[ar] = fmax(x[ar], varargin_2[ar]);
  }

  c_compute_acc_nonlinear_earth_r(u_out_feasible, Theta, Phi, Psi, p, q, r,
    K_p_T, K_p_M, m, I_xx, I_yy, I_zz, l_1, l_2, l_3, l_4, l_z, Cl_alpha,
    Cd_zero, K_Cd, Cm_alpha, Cm_zero, rho, V, S, wing_chord, flight_path_angle,
    Beta, s);
  for (i = 0; i < 6; i++) {
    residuals[i] -= s[i];
  }

  *elapsed_time = toc();
  for (vcol = 0; vcol < 12; vcol++) {
    b_x[vcol] = rtIsNaN(actual_u[vcol]);
  }

  vcol = b_x[0];
  for (ar = 0; ar < 11; ar++) {
    vcol += b_x[ar + 1];
  }

  if (vcol > 0.5) {
    *exitflag = -1.0;
  } else {
    *exitflag = 1.0;
  }
}

/*
 * Testing parameters:
 *  (9e-06, 1.31e-07, 2.35, 0.15, 0.13, 0.2, 0.231, 0.231, 0.39, 0.39, 0, ...
 *  0, 0, 0, 100, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0,...
 *  1, 1, 1, 0,...
 *  1, 100, 400, -50, 500, -60,...
 *  0.01, 0.01, 0.01, 0.1, 0.1, 0.01,...
 *  900, 0, 25, -130, 45, -45, 100, -30, 40, [0 0 -5 0 0 0]', 0, 0, 0, 0.1, 5.18, 0.38, 0.2, ...
 *  -.1, 1.225, 0, 0.55, 0.33, 0, 15, -15, 0, 1e-07,...
 *  0, 0, 0, 0, 0, 100)
 *
 * Arguments    : double K_p_T
 *                double K_p_M
 *                double m
 *                double I_xx
 *                double I_yy
 *                double I_zz
 *                double l_1
 *                double l_2
 *                double l_3
 *                double l_4
 *                double l_z
 *                double Phi
 *                double Theta
 *                double Psi
 *                double Omega_1
 *                double Omega_2
 *                double Omega_3
 *                double Omega_4
 *                double b_1
 *                double b_2
 *                double b_3
 *                double b_4
 *                double g_1
 *                double g_2
 *                double g_3
 *                double g_4
 *                double W_act_motor_const
 *                double W_act_motor_speed
 *                double W_act_tilt_el_const
 *                double W_act_tilt_el_speed
 *                double W_act_tilt_az_const
 *                double W_act_tilt_az_speed
 *                double W_dv_1
 *                double W_dv_2
 *                double W_dv_3
 *                double W_dv_4
 *                double W_dv_5
 *                double W_dv_6
 *                double max_omega
 *                double min_omega
 *                double max_b
 *                double min_b
 *                double max_g
 *                double min_g
 *                double dv[6]
 *                double p
 *                double q
 *                double r
 *                double Cm_zero
 *                double Cl_alpha
 *                double Cd_zero
 *                double K_Cd
 *                double Cm_alpha
 *                double rho
 *                double V
 *                double S
 *                double wing_chord
 *                double flight_path_angle
 *                double max_alpha
 *                double Beta
 *                double gammasq
 *                double desired_motor_value
 *                double desired_el_value
 *                double desired_az_value
 *                double u_out[12]
 *                double residuals[6]
 *                double *elapsed_time
 *                double *N_iterations
 *                double *N_evaluation
 *                double *exitflag
 * Return Type  : void
 */
static void c_WLS_controller_fcn_earth_rf_j(double K_p_T, double K_p_M, double m,
  double I_xx, double I_yy, double I_zz, double l_1, double l_2, double l_3,
  double l_4, double l_z, double Phi, double Theta, double Psi, double Omega_1,
  double Omega_2, double Omega_3, double Omega_4, double b_1, double b_2, double
  b_3, double b_4, double g_1, double g_2, double g_3, double g_4, double
  W_act_motor_const, double W_act_motor_speed, double W_act_tilt_el_const,
  double W_act_tilt_el_speed, double W_act_tilt_az_const, double
  W_act_tilt_az_speed, double W_dv_1, double W_dv_2, double W_dv_3, double
  W_dv_4, double W_dv_5, double W_dv_6, double max_omega, double min_omega,
  double max_b, double min_b, double max_g, double min_g, double dv[6], double p,
  double q, double r, double Cm_zero, double Cl_alpha, double Cd_zero, double
  K_Cd, double Cm_alpha, double rho, double V, double S, double wing_chord,
  double flight_path_angle, double max_alpha, double Beta, double gammasq,
  double desired_motor_value, double desired_el_value, double desired_az_value,
  double u_out[12], double residuals[6], double *elapsed_time, double
  *N_iterations, double *N_evaluation, double *exitflag)
{
  double A[216];
  double Wu[144];
  double B_eval[72];
  double gam_sq[72];
  double Wv[36];
  double b_A[18];
  double b_d[18];
  double actual_u[12];
  double desired_u_scaled[12];
  double max_u_scaled[12];
  double min_u_scaled[12];
  double u_max[12];
  double u_max_constrain[12];
  double u_min_constrain[12];
  double v[6];
  double B_eval_tmp;
  double B_eval_tmp_tmp;
  double W_act_az;
  double W_act_el;
  double W_act_motor;
  double ab_B_eval_tmp;
  double ac_B_eval_tmp;
  double b_B_eval_tmp;
  double bb_B_eval_tmp;
  double c_B_eval_tmp;
  double cb_B_eval_tmp;
  double d;
  double d_B_eval_tmp;
  double db_B_eval_tmp;
  double e_B_eval_tmp;
  double eb_B_eval_tmp;
  double f_B_eval_tmp;
  double fb_B_eval_tmp;
  double g_B_eval_tmp;
  double gain_az;
  double gain_el;
  double gain_motor;
  double gb_B_eval_tmp;
  double h_B_eval_tmp;
  double hb_B_eval_tmp;
  double i_B_eval_tmp;
  double ib_B_eval_tmp;
  double j_B_eval_tmp;
  double jb_B_eval_tmp;
  double k_B_eval_tmp;
  double kb_B_eval_tmp;
  double l_B_eval_tmp;
  double lb_B_eval_tmp;
  double m_B_eval_tmp;
  double mb_B_eval_tmp;
  double n_B_eval_tmp;
  double nb_B_eval_tmp;
  double o_B_eval_tmp;
  double ob_B_eval_tmp;
  double p_B_eval_tmp;
  double pb_B_eval_tmp;
  double q_B_eval_tmp;
  double qb_B_eval_tmp;
  double r_B_eval_tmp;
  double rb_B_eval_tmp;
  double s_B_eval_tmp;
  double sb_B_eval_tmp;
  double t_B_eval_tmp;
  double tb_B_eval_tmp;
  double u_B_eval_tmp;
  double ub_B_eval_tmp;
  double v_B_eval_tmp;
  double vb_B_eval_tmp;
  double w_B_eval_tmp;
  double wb_B_eval_tmp;
  double x_B_eval_tmp;
  double xb_B_eval_tmp;
  double y_B_eval_tmp;
  double yb_B_eval_tmp;
  int A_free_size[2];
  int aoffset;
  int b_N_iterations;
  int b_i;
  int b_trueCount;
  int i;
  int i1;
  int i2;
  int i3;
  int iter;
  int k;
  signed char b_tmp_data[12];
  signed char c_tmp_data[12];
  signed char d_tmp_data[12];
  signed char tmp_data[12];
  bool i_free[12];
  bool exitg1;
  tic();

  /*  Assign geometrical and create variables */
  gain_motor = max_omega / 2.0;
  gain_el = (max_b - min_b) * 3.1415926535897931 / 180.0 / 2.0;
  gain_az = (max_g - min_g) * 3.1415926535897931 / 180.0 / 2.0;

  /*  Identify and evaluate the effectiveness matrix [6,12] */
  B_eval_tmp = cos(Psi);
  b_B_eval_tmp = cos(Phi);
  c_B_eval_tmp = sin(Psi);
  d_B_eval_tmp = sin(Phi);
  e_B_eval_tmp = sin(Theta);
  f_B_eval_tmp = cos(Theta);
  g_B_eval_tmp = cos(g_1);
  h_B_eval_tmp = sin(b_1);
  i_B_eval_tmp = cos(b_1);
  j_B_eval_tmp = sin(g_1);
  k_B_eval_tmp = cos(g_2);
  l_B_eval_tmp = sin(b_2);
  m_B_eval_tmp = cos(b_2);
  n_B_eval_tmp = sin(g_2);
  o_B_eval_tmp = cos(g_3);
  p_B_eval_tmp = sin(b_3);
  q_B_eval_tmp = cos(b_3);
  r_B_eval_tmp = sin(g_3);
  s_B_eval_tmp = cos(g_4);
  t_B_eval_tmp = sin(b_4);
  u_B_eval_tmp = cos(b_4);
  v_B_eval_tmp = sin(g_4);
  w_B_eval_tmp = 2.0 * K_p_T * Omega_1;
  B_eval[0] = -((w_B_eval_tmp * B_eval_tmp * f_B_eval_tmp * h_B_eval_tmp +
                 w_B_eval_tmp * i_B_eval_tmp * g_B_eval_tmp * (d_B_eval_tmp *
    c_B_eval_tmp + b_B_eval_tmp * B_eval_tmp * e_B_eval_tmp)) + 2.0 * K_p_T
                * Omega_1 * cos(b_1) * j_B_eval_tmp * (b_B_eval_tmp *
    c_B_eval_tmp - B_eval_tmp * d_B_eval_tmp * e_B_eval_tmp)) / m;
  x_B_eval_tmp = 2.0 * K_p_T * Omega_2;
  y_B_eval_tmp = sin(Phi) * sin(Psi) + cos(Phi) * cos(Psi) * sin(Theta);
  ab_B_eval_tmp = cos(Phi) * sin(Psi) - cos(Psi) * sin(Phi) * sin(Theta);
  B_eval[6] = -((x_B_eval_tmp * B_eval_tmp * f_B_eval_tmp * l_B_eval_tmp +
                 x_B_eval_tmp * m_B_eval_tmp * k_B_eval_tmp * y_B_eval_tmp) +
                2.0 * K_p_T * Omega_2 * cos(b_2) * n_B_eval_tmp * ab_B_eval_tmp)
    / m;
  bb_B_eval_tmp = 2.0 * K_p_T * Omega_3;
  B_eval[12] = -((bb_B_eval_tmp * B_eval_tmp * f_B_eval_tmp * p_B_eval_tmp +
                  bb_B_eval_tmp * q_B_eval_tmp * o_B_eval_tmp * y_B_eval_tmp) +
                 2.0 * K_p_T * Omega_3 * cos(b_3) * r_B_eval_tmp * ab_B_eval_tmp)
    / m;
  cb_B_eval_tmp = 2.0 * K_p_T * Omega_4;
  B_eval[18] = -((cb_B_eval_tmp * B_eval_tmp * f_B_eval_tmp * t_B_eval_tmp +
                  cb_B_eval_tmp * u_B_eval_tmp * s_B_eval_tmp * y_B_eval_tmp) +
                 2.0 * K_p_T * Omega_4 * cos(b_4) * v_B_eval_tmp * ab_B_eval_tmp)
    / m;
  W_act_motor = Omega_1 * Omega_1;
  db_B_eval_tmp = K_p_T * W_act_motor;
  eb_B_eval_tmp = db_B_eval_tmp * g_B_eval_tmp * h_B_eval_tmp;
  fb_B_eval_tmp = db_B_eval_tmp * h_B_eval_tmp * j_B_eval_tmp;
  B_eval[24] = ((eb_B_eval_tmp * y_B_eval_tmp - db_B_eval_tmp * B_eval_tmp *
                 f_B_eval_tmp * i_B_eval_tmp) + fb_B_eval_tmp * ab_B_eval_tmp) /
    m;
  W_act_el = Omega_2 * Omega_2;
  gb_B_eval_tmp = K_p_T * W_act_el;
  hb_B_eval_tmp = gb_B_eval_tmp * k_B_eval_tmp * l_B_eval_tmp;
  ib_B_eval_tmp = gb_B_eval_tmp * l_B_eval_tmp * n_B_eval_tmp;
  B_eval[30] = ((hb_B_eval_tmp * y_B_eval_tmp - gb_B_eval_tmp * B_eval_tmp *
                 f_B_eval_tmp * m_B_eval_tmp) + ib_B_eval_tmp * ab_B_eval_tmp) /
    m;
  W_act_az = Omega_3 * Omega_3;
  jb_B_eval_tmp = K_p_T * W_act_az;
  kb_B_eval_tmp = jb_B_eval_tmp * o_B_eval_tmp * p_B_eval_tmp;
  lb_B_eval_tmp = jb_B_eval_tmp * p_B_eval_tmp * r_B_eval_tmp;
  B_eval[36] = ((kb_B_eval_tmp * y_B_eval_tmp - jb_B_eval_tmp * B_eval_tmp *
                 f_B_eval_tmp * q_B_eval_tmp) + lb_B_eval_tmp * ab_B_eval_tmp) /
    m;
  B_eval_tmp_tmp = Omega_4 * Omega_4;
  mb_B_eval_tmp = K_p_T * B_eval_tmp_tmp;
  nb_B_eval_tmp = mb_B_eval_tmp * s_B_eval_tmp * t_B_eval_tmp;
  ob_B_eval_tmp = mb_B_eval_tmp * t_B_eval_tmp * v_B_eval_tmp;
  B_eval[42] = ((nb_B_eval_tmp * y_B_eval_tmp - mb_B_eval_tmp * B_eval_tmp *
                 f_B_eval_tmp * u_B_eval_tmp) + ob_B_eval_tmp * ab_B_eval_tmp) /
    m;
  B_eval_tmp = db_B_eval_tmp * i_B_eval_tmp;
  pb_B_eval_tmp = B_eval_tmp * g_B_eval_tmp;
  qb_B_eval_tmp = B_eval_tmp * j_B_eval_tmp;
  B_eval[48] = -(pb_B_eval_tmp * ab_B_eval_tmp - qb_B_eval_tmp * y_B_eval_tmp) /
    m;
  rb_B_eval_tmp = gb_B_eval_tmp * m_B_eval_tmp;
  sb_B_eval_tmp = rb_B_eval_tmp * k_B_eval_tmp;
  tb_B_eval_tmp = rb_B_eval_tmp * n_B_eval_tmp;
  B_eval[54] = -(sb_B_eval_tmp * ab_B_eval_tmp - tb_B_eval_tmp * y_B_eval_tmp) /
    m;
  ub_B_eval_tmp = jb_B_eval_tmp * q_B_eval_tmp;
  vb_B_eval_tmp = ub_B_eval_tmp * o_B_eval_tmp;
  wb_B_eval_tmp = ub_B_eval_tmp * r_B_eval_tmp;
  B_eval[60] = -(vb_B_eval_tmp * ab_B_eval_tmp - wb_B_eval_tmp * y_B_eval_tmp) /
    m;
  xb_B_eval_tmp = mb_B_eval_tmp * u_B_eval_tmp;
  yb_B_eval_tmp = xb_B_eval_tmp * s_B_eval_tmp;
  ac_B_eval_tmp = xb_B_eval_tmp * v_B_eval_tmp;
  B_eval[66] = -(yb_B_eval_tmp * ab_B_eval_tmp - ac_B_eval_tmp * y_B_eval_tmp) /
    m;
  B_eval[1] = ((2.0 * K_p_T * Omega_1 * cos(b_1) * cos(g_1) * (cos(Psi) * sin
    (Phi) - cos(Phi) * sin(Psi) * e_B_eval_tmp) - w_B_eval_tmp * f_B_eval_tmp *
                c_B_eval_tmp * h_B_eval_tmp) + 2.0 * K_p_T * Omega_1 * cos(b_1) *
               sin(g_1) * (cos(Phi) * cos(Psi) + sin(Phi) * sin(Psi) *
    e_B_eval_tmp)) / m;
  e_B_eval_tmp = cos(Psi) * sin(Phi) - cos(Phi) * sin(Psi) * sin(Theta);
  y_B_eval_tmp = cos(Phi) * cos(Psi) + sin(Phi) * sin(Psi) * sin(Theta);
  B_eval[7] = ((2.0 * K_p_T * Omega_2 * cos(b_2) * cos(g_2) * e_B_eval_tmp -
                x_B_eval_tmp * f_B_eval_tmp * c_B_eval_tmp * l_B_eval_tmp) + 2.0
               * K_p_T * Omega_2 * cos(b_2) * sin(g_2) * y_B_eval_tmp) / m;
  B_eval[13] = ((2.0 * K_p_T * Omega_3 * cos(b_3) * cos(g_3) * e_B_eval_tmp -
                 bb_B_eval_tmp * f_B_eval_tmp * c_B_eval_tmp * p_B_eval_tmp) +
                2.0 * K_p_T * Omega_3 * cos(b_3) * sin(g_3) * y_B_eval_tmp) / m;
  B_eval[19] = ((2.0 * K_p_T * Omega_4 * cos(b_4) * cos(g_4) * e_B_eval_tmp -
                 cb_B_eval_tmp * f_B_eval_tmp * c_B_eval_tmp * t_B_eval_tmp) +
                2.0 * K_p_T * Omega_4 * cos(b_4) * sin(g_4) * y_B_eval_tmp) / m;
  ab_B_eval_tmp = db_B_eval_tmp * f_B_eval_tmp;
  B_eval[25] = -((ab_B_eval_tmp * c_B_eval_tmp * i_B_eval_tmp + eb_B_eval_tmp *
                  e_B_eval_tmp) + fb_B_eval_tmp * y_B_eval_tmp) / m;
  eb_B_eval_tmp = gb_B_eval_tmp * f_B_eval_tmp;
  B_eval[31] = -((eb_B_eval_tmp * c_B_eval_tmp * m_B_eval_tmp + hb_B_eval_tmp *
                  e_B_eval_tmp) + ib_B_eval_tmp * y_B_eval_tmp) / m;
  fb_B_eval_tmp = jb_B_eval_tmp * f_B_eval_tmp;
  B_eval[37] = -((fb_B_eval_tmp * c_B_eval_tmp * q_B_eval_tmp + kb_B_eval_tmp *
                  e_B_eval_tmp) + lb_B_eval_tmp * y_B_eval_tmp) / m;
  hb_B_eval_tmp = mb_B_eval_tmp * f_B_eval_tmp;
  B_eval[43] = -((hb_B_eval_tmp * c_B_eval_tmp * u_B_eval_tmp + nb_B_eval_tmp *
                  e_B_eval_tmp) + ob_B_eval_tmp * y_B_eval_tmp) / m;
  B_eval[49] = (pb_B_eval_tmp * y_B_eval_tmp - qb_B_eval_tmp * e_B_eval_tmp) / m;
  B_eval[55] = (sb_B_eval_tmp * y_B_eval_tmp - tb_B_eval_tmp * e_B_eval_tmp) / m;
  B_eval[61] = (vb_B_eval_tmp * y_B_eval_tmp - wb_B_eval_tmp * e_B_eval_tmp) / m;
  B_eval[67] = (yb_B_eval_tmp * y_B_eval_tmp - ac_B_eval_tmp * e_B_eval_tmp) / m;
  B_eval[2] = ((w_B_eval_tmp * h_B_eval_tmp * e_B_eval_tmp + 2.0 * K_p_T
                * Omega_1 * cos(Theta) * d_B_eval_tmp * i_B_eval_tmp *
                j_B_eval_tmp) - w_B_eval_tmp * b_B_eval_tmp * f_B_eval_tmp *
               i_B_eval_tmp * g_B_eval_tmp) / m;
  B_eval[8] = ((x_B_eval_tmp * l_B_eval_tmp * e_B_eval_tmp + 2.0 * K_p_T
                * Omega_2 * cos(Theta) * d_B_eval_tmp * m_B_eval_tmp *
                n_B_eval_tmp) - x_B_eval_tmp * b_B_eval_tmp * f_B_eval_tmp *
               m_B_eval_tmp * k_B_eval_tmp) / m;
  B_eval[14] = ((bb_B_eval_tmp * p_B_eval_tmp * e_B_eval_tmp + 2.0 * K_p_T
                 * Omega_3 * cos(Theta) * d_B_eval_tmp * q_B_eval_tmp *
                 r_B_eval_tmp) - bb_B_eval_tmp * b_B_eval_tmp * f_B_eval_tmp *
                q_B_eval_tmp * o_B_eval_tmp) / m;
  B_eval[20] = ((cb_B_eval_tmp * t_B_eval_tmp * e_B_eval_tmp + 2.0 * K_p_T
                 * Omega_4 * cos(Theta) * d_B_eval_tmp * u_B_eval_tmp *
                 v_B_eval_tmp) - cb_B_eval_tmp * b_B_eval_tmp * f_B_eval_tmp *
                u_B_eval_tmp * s_B_eval_tmp) / m;
  c_B_eval_tmp = db_B_eval_tmp * b_B_eval_tmp * f_B_eval_tmp;
  y_B_eval_tmp = ab_B_eval_tmp * d_B_eval_tmp;
  B_eval[26] = ((B_eval_tmp * e_B_eval_tmp + c_B_eval_tmp * g_B_eval_tmp *
                 h_B_eval_tmp) - y_B_eval_tmp * h_B_eval_tmp * j_B_eval_tmp) / m;
  B_eval_tmp = gb_B_eval_tmp * b_B_eval_tmp * f_B_eval_tmp;
  ab_B_eval_tmp = eb_B_eval_tmp * d_B_eval_tmp;
  B_eval[32] = ((rb_B_eval_tmp * e_B_eval_tmp + B_eval_tmp * k_B_eval_tmp *
                 l_B_eval_tmp) - ab_B_eval_tmp * l_B_eval_tmp * n_B_eval_tmp) /
    m;
  eb_B_eval_tmp = jb_B_eval_tmp * b_B_eval_tmp * f_B_eval_tmp;
  fb_B_eval_tmp *= d_B_eval_tmp;
  B_eval[38] = ((ub_B_eval_tmp * e_B_eval_tmp + eb_B_eval_tmp * o_B_eval_tmp *
                 p_B_eval_tmp) - fb_B_eval_tmp * p_B_eval_tmp * r_B_eval_tmp) /
    m;
  b_B_eval_tmp = mb_B_eval_tmp * b_B_eval_tmp * f_B_eval_tmp;
  d_B_eval_tmp *= hb_B_eval_tmp;
  B_eval[44] = ((xb_B_eval_tmp * e_B_eval_tmp + b_B_eval_tmp * s_B_eval_tmp *
                 t_B_eval_tmp) - d_B_eval_tmp * t_B_eval_tmp * v_B_eval_tmp) / m;
  B_eval[50] = (c_B_eval_tmp * i_B_eval_tmp * j_B_eval_tmp + y_B_eval_tmp *
                i_B_eval_tmp * g_B_eval_tmp) / m;
  B_eval[56] = (B_eval_tmp * m_B_eval_tmp * n_B_eval_tmp + ab_B_eval_tmp *
                m_B_eval_tmp * k_B_eval_tmp) / m;
  B_eval[62] = (eb_B_eval_tmp * q_B_eval_tmp * r_B_eval_tmp + fb_B_eval_tmp *
                q_B_eval_tmp * o_B_eval_tmp) / m;
  B_eval[68] = (b_B_eval_tmp * u_B_eval_tmp * v_B_eval_tmp + d_B_eval_tmp *
                u_B_eval_tmp * s_B_eval_tmp) / m;
  B_eval_tmp = w_B_eval_tmp * l_z;
  b_B_eval_tmp = 2.0 * K_p_M * Omega_1;
  c_B_eval_tmp = w_B_eval_tmp * l_1;
  B_eval[3] = ((b_B_eval_tmp * h_B_eval_tmp + B_eval_tmp * i_B_eval_tmp *
                j_B_eval_tmp) + c_B_eval_tmp * i_B_eval_tmp * g_B_eval_tmp) /
    I_xx;
  d_B_eval_tmp = x_B_eval_tmp * l_z;
  e_B_eval_tmp = 2.0 * K_p_M * Omega_2;
  f_B_eval_tmp = x_B_eval_tmp * l_1;
  B_eval[9] = -((e_B_eval_tmp * l_B_eval_tmp - d_B_eval_tmp * m_B_eval_tmp *
                 n_B_eval_tmp) + f_B_eval_tmp * m_B_eval_tmp * k_B_eval_tmp) /
    I_xx;
  y_B_eval_tmp = 2.0 * K_p_M * Omega_3;
  ab_B_eval_tmp = bb_B_eval_tmp * l_z;
  eb_B_eval_tmp = bb_B_eval_tmp * l_2;
  B_eval[15] = ((y_B_eval_tmp * p_B_eval_tmp + ab_B_eval_tmp * q_B_eval_tmp *
                 r_B_eval_tmp) - eb_B_eval_tmp * q_B_eval_tmp * o_B_eval_tmp) /
    I_xx;
  fb_B_eval_tmp = cb_B_eval_tmp * l_z;
  hb_B_eval_tmp = 2.0 * K_p_M * Omega_4;
  ib_B_eval_tmp = cb_B_eval_tmp * l_2;
  B_eval[21] = ((fb_B_eval_tmp * u_B_eval_tmp * v_B_eval_tmp - hb_B_eval_tmp *
                 t_B_eval_tmp) + ib_B_eval_tmp * u_B_eval_tmp * s_B_eval_tmp) /
    I_xx;
  kb_B_eval_tmp = db_B_eval_tmp * l_z;
  lb_B_eval_tmp = db_B_eval_tmp * l_1;
  nb_B_eval_tmp = K_p_M * W_act_motor;
  ob_B_eval_tmp = nb_B_eval_tmp * i_B_eval_tmp;
  B_eval[27] = -((lb_B_eval_tmp * g_B_eval_tmp * h_B_eval_tmp - ob_B_eval_tmp) +
                 kb_B_eval_tmp * h_B_eval_tmp * j_B_eval_tmp) / I_xx;
  pb_B_eval_tmp = gb_B_eval_tmp * l_z;
  qb_B_eval_tmp = gb_B_eval_tmp * l_1;
  rb_B_eval_tmp = K_p_M * W_act_el;
  sb_B_eval_tmp = rb_B_eval_tmp * m_B_eval_tmp;
  B_eval[33] = -((sb_B_eval_tmp - qb_B_eval_tmp * k_B_eval_tmp * l_B_eval_tmp) +
                 pb_B_eval_tmp * l_B_eval_tmp * n_B_eval_tmp) / I_xx;
  tb_B_eval_tmp = jb_B_eval_tmp * l_z;
  ub_B_eval_tmp = jb_B_eval_tmp * l_2;
  vb_B_eval_tmp = K_p_M * W_act_az;
  wb_B_eval_tmp = vb_B_eval_tmp * q_B_eval_tmp;
  B_eval[39] = ((wb_B_eval_tmp + ub_B_eval_tmp * o_B_eval_tmp * p_B_eval_tmp) -
                tb_B_eval_tmp * p_B_eval_tmp * r_B_eval_tmp) / I_xx;
  xb_B_eval_tmp = mb_B_eval_tmp * l_z;
  yb_B_eval_tmp = mb_B_eval_tmp * l_2;
  ac_B_eval_tmp = K_p_M * B_eval_tmp_tmp;
  W_act_motor = ac_B_eval_tmp * u_B_eval_tmp;
  B_eval[45] = -((W_act_motor + yb_B_eval_tmp * s_B_eval_tmp * t_B_eval_tmp) +
                 xb_B_eval_tmp * t_B_eval_tmp * v_B_eval_tmp) / I_xx;
  kb_B_eval_tmp *= i_B_eval_tmp;
  lb_B_eval_tmp *= i_B_eval_tmp;
  B_eval[51] = (kb_B_eval_tmp * g_B_eval_tmp - lb_B_eval_tmp * j_B_eval_tmp) /
    I_xx;
  pb_B_eval_tmp *= m_B_eval_tmp;
  qb_B_eval_tmp *= m_B_eval_tmp;
  B_eval[57] = (pb_B_eval_tmp * k_B_eval_tmp + qb_B_eval_tmp * n_B_eval_tmp) /
    I_xx;
  tb_B_eval_tmp *= q_B_eval_tmp;
  ub_B_eval_tmp *= q_B_eval_tmp;
  B_eval[63] = (tb_B_eval_tmp * o_B_eval_tmp + ub_B_eval_tmp * r_B_eval_tmp) /
    I_xx;
  xb_B_eval_tmp *= u_B_eval_tmp;
  yb_B_eval_tmp *= u_B_eval_tmp;
  B_eval[69] = (xb_B_eval_tmp * s_B_eval_tmp - yb_B_eval_tmp * v_B_eval_tmp) /
    I_xx;
  B_eval[4] = ((B_eval_tmp * h_B_eval_tmp - b_B_eval_tmp * i_B_eval_tmp *
                j_B_eval_tmp) + w_B_eval_tmp * l_4 * i_B_eval_tmp * g_B_eval_tmp)
    / I_yy;
  B_eval[10] = ((d_B_eval_tmp * l_B_eval_tmp + e_B_eval_tmp * m_B_eval_tmp *
                 n_B_eval_tmp) + x_B_eval_tmp * l_4 * m_B_eval_tmp *
                k_B_eval_tmp) / I_yy;
  B_eval[16] = -((y_B_eval_tmp * q_B_eval_tmp * r_B_eval_tmp - ab_B_eval_tmp *
                  p_B_eval_tmp) + bb_B_eval_tmp * l_3 * q_B_eval_tmp *
                 o_B_eval_tmp) / I_yy;
  B_eval[22] = ((fb_B_eval_tmp * t_B_eval_tmp + hb_B_eval_tmp * u_B_eval_tmp *
                 v_B_eval_tmp) - cb_B_eval_tmp * l_3 * u_B_eval_tmp *
                s_B_eval_tmp) / I_yy;
  B_eval_tmp = db_B_eval_tmp * l_4;
  B_eval[28] = ((nb_B_eval_tmp * h_B_eval_tmp * j_B_eval_tmp + kb_B_eval_tmp) -
                B_eval_tmp * g_B_eval_tmp * h_B_eval_tmp) / I_yy;
  b_B_eval_tmp = gb_B_eval_tmp * l_4;
  B_eval[34] = -((rb_B_eval_tmp * l_B_eval_tmp * n_B_eval_tmp - pb_B_eval_tmp) +
                 b_B_eval_tmp * k_B_eval_tmp * l_B_eval_tmp) / I_yy;
  d_B_eval_tmp = jb_B_eval_tmp * l_3;
  B_eval[40] = ((vb_B_eval_tmp * p_B_eval_tmp * r_B_eval_tmp + tb_B_eval_tmp) +
                d_B_eval_tmp * o_B_eval_tmp * p_B_eval_tmp) / I_yy;
  e_B_eval_tmp = mb_B_eval_tmp * l_3;
  B_eval[46] = ((xb_B_eval_tmp - ac_B_eval_tmp * t_B_eval_tmp * v_B_eval_tmp) +
                e_B_eval_tmp * s_B_eval_tmp * t_B_eval_tmp) / I_yy;
  i_B_eval_tmp *= B_eval_tmp;
  B_eval[52] = -(ob_B_eval_tmp * g_B_eval_tmp + i_B_eval_tmp * j_B_eval_tmp) /
    I_yy;
  m_B_eval_tmp *= b_B_eval_tmp;
  B_eval[58] = (sb_B_eval_tmp * k_B_eval_tmp - m_B_eval_tmp * n_B_eval_tmp) /
    I_yy;
  q_B_eval_tmp *= d_B_eval_tmp;
  B_eval[64] = -(wb_B_eval_tmp * o_B_eval_tmp - q_B_eval_tmp * r_B_eval_tmp) /
    I_yy;
  u_B_eval_tmp *= e_B_eval_tmp;
  B_eval[70] = (W_act_motor * s_B_eval_tmp + u_B_eval_tmp * v_B_eval_tmp) / I_yy;
  B_eval[5] = ((2.0 * K_p_M * Omega_1 * cos(b_1) * g_B_eval_tmp - c_B_eval_tmp *
                h_B_eval_tmp) + 2.0 * K_p_T * Omega_1 * l_4 * cos(b_1) *
               j_B_eval_tmp) / I_zz;
  B_eval[11] = ((f_B_eval_tmp * l_B_eval_tmp - 2.0 * K_p_M * Omega_2 * cos(b_2) *
                 k_B_eval_tmp) + 2.0 * K_p_T * Omega_2 * l_4 * cos(b_2) *
                n_B_eval_tmp) / I_zz;
  B_eval[17] = ((eb_B_eval_tmp * p_B_eval_tmp + 2.0 * K_p_M * Omega_3 * cos(b_3)
                 * o_B_eval_tmp) - 2.0 * K_p_T * Omega_3 * l_3 * cos(b_3) *
                r_B_eval_tmp) / I_zz;
  B_eval[23] = -((ib_B_eval_tmp * t_B_eval_tmp + 2.0 * K_p_M * Omega_4 * cos(b_4)
                  * s_B_eval_tmp) + 2.0 * K_p_T * Omega_4 * l_3 * cos(b_4) *
                 v_B_eval_tmp) / I_zz;
  B_eval[29] = -((lb_B_eval_tmp + nb_B_eval_tmp * g_B_eval_tmp * h_B_eval_tmp) +
                 B_eval_tmp * h_B_eval_tmp * j_B_eval_tmp) / I_zz;
  B_eval[35] = ((qb_B_eval_tmp + rb_B_eval_tmp * k_B_eval_tmp * l_B_eval_tmp) -
                b_B_eval_tmp * l_B_eval_tmp * n_B_eval_tmp) / I_zz;
  B_eval[41] = ((ub_B_eval_tmp - vb_B_eval_tmp * o_B_eval_tmp * p_B_eval_tmp) +
                d_B_eval_tmp * p_B_eval_tmp * r_B_eval_tmp) / I_zz;
  B_eval[47] = ((ac_B_eval_tmp * s_B_eval_tmp * t_B_eval_tmp - yb_B_eval_tmp) +
                e_B_eval_tmp * t_B_eval_tmp * v_B_eval_tmp) / I_zz;
  B_eval[53] = -(ob_B_eval_tmp * j_B_eval_tmp - i_B_eval_tmp * g_B_eval_tmp) /
    I_zz;
  B_eval[59] = (sb_B_eval_tmp * n_B_eval_tmp + m_B_eval_tmp * k_B_eval_tmp) /
    I_zz;
  B_eval[65] = -(wb_B_eval_tmp * r_B_eval_tmp + q_B_eval_tmp * o_B_eval_tmp) /
    I_zz;
  B_eval[71] = (W_act_motor * v_B_eval_tmp - u_B_eval_tmp * s_B_eval_tmp) / I_zz;
  actual_u[0] = Omega_1;
  actual_u[1] = Omega_2;
  actual_u[2] = Omega_3;
  actual_u[3] = Omega_4;
  actual_u[4] = b_1;
  actual_u[5] = b_2;
  actual_u[6] = b_3;
  actual_u[7] = b_4;
  actual_u[8] = g_1;
  actual_u[9] = g_2;
  actual_u[10] = g_3;
  actual_u[11] = g_4;

  /*  Apply gains for the normalization of the effectiveness matrix: */
  c_compute_acc_nonlinear_earth_r(actual_u, Theta, Phi, Psi, p, q, r, K_p_T,
    K_p_M, m, I_xx, I_yy, I_zz, l_1, l_2, l_3, l_4, l_z, Cl_alpha, Cd_zero, K_Cd,
    Cm_alpha, Cm_zero, rho, V, S, wing_chord, flight_path_angle, Beta, residuals);
  for (i = 0; i < 6; i++) {
    W_act_el = dv[i];
    residuals[i] += W_act_el;
    W_act_motor = 0.0;
    for (k = 0; k < 12; k++) {
      W_act_motor += B_eval[i + 6 * k] * actual_u[k];
    }

    W_act_el += W_act_motor;
    dv[i] = W_act_el;
  }

  for (i = 0; i < 4; i++) {
    for (k = 0; k < 6; k++) {
      aoffset = k + 6 * i;
      B_eval[aoffset] *= gain_motor;
      aoffset = k + 6 * (i + 4);
      B_eval[aoffset] *= gain_el;
      aoffset = k + 6 * (i + 8);
      B_eval[aoffset] *= gain_az;
    }
  }

  /*  Apply WLS algorithm: */
  /* Build the max and minimum actuator array:  */
  u_max[0] = max_omega;
  u_max[1] = max_omega;
  u_max[2] = max_omega;
  u_max[3] = max_omega;
  u_max[4] = max_b;
  u_max[5] = max_b;
  u_max[6] = max_b;
  u_max[7] = max_b;
  u_max[8] = max_g;
  u_max[9] = max_g;
  u_max[10] = max_g;
  u_max[11] = max_g;
  u_out[0] = min_omega;
  u_out[1] = min_omega;
  u_out[2] = min_omega;
  u_out[3] = min_omega;
  u_out[4] = min_b;
  u_out[5] = min_b;
  u_out[6] = min_b;
  u_out[7] = min_b;
  u_out[8] = min_g;
  u_out[9] = min_g;
  u_out[10] = min_g;
  u_out[11] = min_g;
  for (i = 0; i < 8; i++) {
    u_max[i + 4] = u_max[i + 4] * 3.1415926535897931 / 180.0;
    u_out[i + 4] = u_out[i + 4] * 3.1415926535897931 / 180.0;
  }

  /* Comput now the delta to take from the actual actuator position to the desired one:  */
  W_act_motor = desired_motor_value / gain_motor;
  desired_u_scaled[0] = W_act_motor;
  desired_u_scaled[1] = W_act_motor;
  desired_u_scaled[2] = W_act_motor;
  desired_u_scaled[3] = W_act_motor;
  W_act_motor = desired_el_value / gain_el;
  desired_u_scaled[4] = W_act_motor;
  W_act_el = desired_az_value / gain_az;
  desired_u_scaled[8] = W_act_el;
  desired_u_scaled[5] = W_act_motor;
  desired_u_scaled[9] = W_act_el;
  desired_u_scaled[6] = W_act_motor;
  desired_u_scaled[10] = W_act_el;
  desired_u_scaled[7] = W_act_motor;
  desired_u_scaled[11] = W_act_el;

  /* Compute the maximum and minimum control input value for each channel:  */
  /*  Motor */
  memcpy(&u_max_constrain[0], &u_max[0], 12U * sizeof(double));
  memcpy(&u_min_constrain[0], &u_out[0], 12U * sizeof(double));
  if (max_alpha > 3.0) {
    W_act_motor = max_alpha * 3.1415926535897931 / 180.0;
    for (i = 0; i < 8; i++) {
      W_act_el = actual_u[i + 4];
      u_max_constrain[i + 4] = W_act_el + W_act_motor;
      u_min_constrain[i + 4] = W_act_el - W_act_motor;
    }
  }

  /*  Elevator */
  for (k = 0; k < 12; k++) {
    max_u_scaled[k] = fmin(u_max[k], u_max_constrain[k]);
    min_u_scaled[k] = fmax(u_out[k], u_min_constrain[k]);
  }

  max_u_scaled[0] /= gain_motor;
  min_u_scaled[0] /= gain_motor;
  max_u_scaled[4] /= gain_el;
  min_u_scaled[4] /= gain_el;
  max_u_scaled[8] /= gain_az;
  min_u_scaled[8] /= gain_az;
  max_u_scaled[1] /= gain_motor;
  min_u_scaled[1] /= gain_motor;
  max_u_scaled[5] /= gain_el;
  min_u_scaled[5] /= gain_el;
  max_u_scaled[9] /= gain_az;
  min_u_scaled[9] /= gain_az;
  max_u_scaled[2] /= gain_motor;
  min_u_scaled[2] /= gain_motor;
  max_u_scaled[6] /= gain_el;
  min_u_scaled[6] /= gain_el;
  max_u_scaled[10] /= gain_az;
  min_u_scaled[10] /= gain_az;
  max_u_scaled[3] /= gain_motor;
  min_u_scaled[3] /= gain_motor;
  max_u_scaled[7] /= gain_el;
  min_u_scaled[7] /= gain_el;
  max_u_scaled[11] /= gain_az;
  min_u_scaled[11] /= gain_az;

  /* Weighting matrix for the control objective and pseudo control: */
  W_act_motor = fmax(1.0, W_act_motor_const + W_act_motor_speed * V);
  W_act_el = fmax(1.0, W_act_tilt_el_const + W_act_tilt_el_speed * V);
  W_act_az = fmax(1.0, W_act_tilt_az_const + W_act_tilt_az_speed * V);
  v[0] = W_dv_1;
  v[1] = W_dv_2;
  v[2] = W_dv_3;
  v[3] = W_dv_4;
  v[4] = W_dv_5;
  v[5] = W_dv_6;
  memset(&Wv[0], 0, 36U * sizeof(double));
  for (aoffset = 0; aoffset < 6; aoffset++) {
    Wv[aoffset + 6 * aoffset] = v[aoffset];
  }

  actual_u[0] = W_act_motor;
  actual_u[1] = W_act_motor;
  actual_u[2] = W_act_motor;
  actual_u[3] = W_act_motor;
  actual_u[4] = W_act_el;
  actual_u[5] = W_act_el;
  actual_u[6] = W_act_el;
  actual_u[7] = W_act_el;
  actual_u[8] = W_act_az;
  actual_u[9] = W_act_az;
  actual_u[10] = W_act_az;
  actual_u[11] = W_act_az;
  memset(&Wu[0], 0, 144U * sizeof(double));
  for (aoffset = 0; aoffset < 12; aoffset++) {
    Wu[aoffset + 12 * aoffset] = actual_u[aoffset];
  }

  /*  WLS_ALLOC - Control allocation using weighted least squares. */
  /*  */
  /*   [u,W,iter] = wls_alloc(B,v,umin,umax,[Wv,Wu,ud,gamma,u0,W0,imax]) */
  /*  */
  /*  Solves the weighted, bounded least-squares problem */
  /*  */
  /*    min ||Wu(u-ud)||^2 + gamma ||Wv(Bu-v)||^2 */
  /*  */
  /*    subj. to  umin <= u <= umax */
  /*  */
  /*  using an active set method. */
  /*  */
  /*   Inputs: */
  /*   ------- */
  /*  B     control effectiveness matrix (k x m) */
  /*  v     commanded virtual control (k x 1) */
  /*  umin  lower position limits (m x 1) */
  /*  umax  upper position limits (m x 1) */
  /*  Wv    virtual control weighting matrix (k x k) [I] */
  /*  Wu    control weighting matrix (m x m) [I] */
  /*  ud    desired control (m x 1) [0] */
  /*  gamma weight (scalar) [1e6] */
  /*  u0    initial point (m x 1) */
  /*  W0    initial working set (m x 1) [empty] */
  /*  imax  max no. of iterations [100] */
  /*   */
  /*   Outputs: */
  /*   ------- */
  /*  u     optimal control */
  /*  W     optimal active set */
  /*  iter  no. of iterations (= no. of changes in the working set + 1) */
  /*  */
  /*                             0 if u_i not saturated */
  /*  Working set syntax: W_i = -1 if u_i = umin_i */
  /*                            +1 if u_i = umax_i */
  /*  */
  /*  See also: WLSC_ALLOC, IP_ALLOC, FXP_ALLOC, QP_SIM. */
  memset(&u_max_constrain[0], 0, 12U * sizeof(double));
  memset(&u_out[0], 0, 12U * sizeof(double));

  /*  Number of variables */
  /*  Set default values of optional arguments */
  W_act_motor = sqrt(gammasq * gammasq);

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4) \
 private(i2,d,i3)

  for (i1 = 0; i1 < 6; i1++) {
    for (i3 = 0; i3 < 12; i3++) {
      d = 0.0;
      for (i2 = 0; i2 < 6; i2++) {
        d += W_act_motor * Wv[i1 + 6 * i2] * B_eval[i2 + 6 * i3];
      }

      gam_sq[i1 + 6 * i3] = d;
    }
  }

  for (i = 0; i < 12; i++) {
    for (k = 0; k < 6; k++) {
      A[k + 18 * i] = gam_sq[k + 6 * i];
    }

    memcpy(&A[i * 18 + 6], &Wu[i * 12], 12U * sizeof(double));
  }

  /*  Initial residual. */
  for (i = 0; i < 6; i++) {
    W_act_el = 0.0;
    for (k = 0; k < 6; k++) {
      W_act_el += W_act_motor * Wv[i + 6 * k] * dv[k];
    }

    v[i] = W_act_el;
  }

  for (i = 0; i < 12; i++) {
    W_act_el = 0.0;
    for (k = 0; k < 12; k++) {
      W_act_el += Wu[i + 12 * k] * desired_u_scaled[k];
    }

    u_min_constrain[i] = W_act_el;
  }

  for (i = 0; i < 6; i++) {
    b_d[i] = v[i];
  }

  memcpy(&b_d[6], &u_min_constrain[0], 12U * sizeof(double));
  for (i = 0; i < 18; i++) {
    W_act_el = 0.0;
    for (k = 0; k < 12; k++) {
      W_act_el += A[i + 18 * k] * 0.0;
    }

    b_d[i] -= W_act_el;
  }

  /*  Determine indeces of free variables. */
  for (b_i = 0; b_i < 12; b_i++) {
    i_free[b_i] = true;
  }

  /*  Iterate until optimum is found or maximum number of iterations */
  /*  is reached. */
  b_N_iterations = 1;
  iter = 0;
  exitg1 = false;
  while ((!exitg1) && (iter < 100)) {
    double A_free_data[216];
    int trueCount;
    bool x_data[12];
    bool exitg2;
    bool y;
    b_N_iterations = iter + 1;

    /*  ---------------------------------------- */
    /*   Compute optimal perturbation vector p. */
    /*  ---------------------------------------- */
    /*  Eliminate saturated variables. */
    trueCount = 0;
    aoffset = 0;
    for (b_i = 0; b_i < 12; b_i++) {
      if (i_free[b_i]) {
        trueCount++;
        tmp_data[aoffset] = (signed char)(b_i + 1);
        aoffset++;
      }
    }

    A_free_size[0] = 18;
    A_free_size[1] = trueCount;
    for (i = 0; i < trueCount; i++) {
      for (k = 0; k < 18; k++) {
        A_free_data[k + 18 * i] = A[k + 18 * (tmp_data[i] - 1)];
      }
    }

    /*  Solve the reduced optimization problem for free variables. */
    mldivide(A_free_data, A_free_size, b_d, u_min_constrain, &aoffset);

    /*  Zero all perturbations corresponding to active constraints. */
    /*  Insert perturbations from p_free into free the variables. */
    aoffset = 0;

    /*  ---------------------------- */
    /*   Is the new point feasible? */
    /*  ---------------------------- */
    b_trueCount = 0;
    k = 0;
    for (b_i = 0; b_i < 12; b_i++) {
      W_act_el = 0.0;
      u_max[b_i] = 0.0;
      y = i_free[b_i];
      if (y) {
        W_act_el = u_min_constrain[aoffset];
        u_max[b_i] = u_min_constrain[aoffset];
        aoffset++;
      }

      actual_u[b_i] = u_out[b_i] + W_act_el;
      if (y) {
        b_trueCount++;
        b_tmp_data[k] = (signed char)(b_i + 1);
        k++;
      }
    }

    for (i = 0; i < b_trueCount; i++) {
      aoffset = b_tmp_data[i] - 1;
      W_act_motor = actual_u[aoffset];
      x_data[i] = ((W_act_motor < min_u_scaled[aoffset]) || (W_act_motor >
        max_u_scaled[aoffset]));
    }

    y = false;
    aoffset = 1;
    exitg2 = false;
    while ((!exitg2) && (aoffset <= b_trueCount)) {
      if (x_data[aoffset - 1]) {
        y = true;
        exitg2 = true;
      } else {
        aoffset++;
      }
    }

    if (!y) {
      /*  ---------------------------- */
      /*   Yes, check for optimality. */
      /*  ---------------------------- */
      /*  Update point and residual. */
      memcpy(&u_out[0], &actual_u[0], 12U * sizeof(double));
      memset(&b_A[0], 0, 18U * sizeof(double));
      for (k = 0; k < trueCount; k++) {
        aoffset = k * 18;
        for (b_i = 0; b_i < 18; b_i++) {
          i = aoffset + b_i;
          b_A[b_i] += A[i % 18 + 18 * (tmp_data[i / 18] - 1)] *
            u_min_constrain[k];
        }
      }

      for (i = 0; i < 18; i++) {
        b_d[i] -= b_A[i];
      }

      /*  Compute Lagrangian multipliers. */
      for (i = 0; i < 12; i++) {
        W_act_el = 0.0;
        for (k = 0; k < 18; k++) {
          W_act_el += A[k + 18 * i] * b_d[k];
        }

        actual_u[i] = u_max_constrain[i] * W_act_el;
      }

      /*  Are all lambda non-negative? */
      y = true;
      k = 0;
      exitg2 = false;
      while ((!exitg2) && (k < 12)) {
        if (!(actual_u[k] >= -2.2204460492503131E-16)) {
          y = false;
          exitg2 = true;
        } else {
          k++;
        }
      }

      if (y) {
        /*  / ------------------------ \ */
        /*  | Optimum found, bail out. | */
        /*  \ ------------------------ / */
        exitg1 = true;
      } else {
        /*  -------------------------------------------------- */
        /*   Optimum not found, remove one active constraint. */
        /*  -------------------------------------------------- */
        /*  Remove constraint with most negative lambda from the */
        /*  working set. */
        minimum(actual_u, &W_act_motor, &b_trueCount);
        u_max_constrain[b_trueCount - 1] = 0.0;
        i_free[b_trueCount - 1] = true;
        iter++;
      }
    } else {
      bool bv[12];

      /*  --------------------------------------- */
      /*   No, find primary bounding constraint. */
      /*  --------------------------------------- */
      /*  Compute distances to the different boundaries. Since alpha < 1 */
      /*  is the maximum step length, initiate with ones. */
      b_trueCount = 0;
      aoffset = 0;
      for (b_i = 0; b_i < 12; b_i++) {
        bool b;
        actual_u[b_i] = 1.0;
        W_act_el = u_max[b_i];
        y = (W_act_el < 0.0);
        x_data[b_i] = y;
        bv[b_i] = (W_act_el > 0.0);
        b = i_free[b_i];
        if (b && y) {
          b_trueCount++;
          c_tmp_data[aoffset] = (signed char)(b_i + 1);
          aoffset++;
        }
      }

      for (i = 0; i < b_trueCount; i++) {
        k = c_tmp_data[i] - 1;
        desired_u_scaled[i] = (min_u_scaled[k] - u_out[k]) / u_max[k];
      }

      aoffset = 0;
      b_trueCount = 0;
      k = 0;
      for (b_i = 0; b_i < 12; b_i++) {
        y = i_free[b_i];
        if (y && x_data[b_i]) {
          actual_u[b_i] = desired_u_scaled[aoffset];
          aoffset++;
        }

        if (y && bv[b_i]) {
          b_trueCount++;
          d_tmp_data[k] = (signed char)(b_i + 1);
          k++;
        }
      }

      for (i = 0; i < b_trueCount; i++) {
        k = d_tmp_data[i] - 1;
        desired_u_scaled[i] = (max_u_scaled[k] - u_out[k]) / u_max[k];
      }

      aoffset = 0;
      for (b_i = 0; b_i < 12; b_i++) {
        if (i_free[b_i] && bv[b_i]) {
          actual_u[b_i] = desired_u_scaled[aoffset];
          aoffset++;
        }
      }

      /*  Proportion of p to travel */
      minimum(actual_u, &W_act_motor, &b_trueCount);

      /*  Update point and residual. */
      for (i = 0; i < 12; i++) {
        u_out[i] += W_act_motor * u_max[i];
      }

      aoffset = 18 * trueCount;
      for (i = 0; i < aoffset; i++) {
        A_free_data[i] *= W_act_motor;
      }

      memset(&b_A[0], 0, 18U * sizeof(double));
      for (k = 0; k < trueCount; k++) {
        aoffset = k * 18;
        for (b_i = 0; b_i < 18; b_i++) {
          b_A[b_i] += A_free_data[aoffset + b_i] * u_min_constrain[k];
        }
      }

      for (i = 0; i < 18; i++) {
        b_d[i] -= b_A[i];
      }

      /*  Add corresponding constraint to working set. */
      W_act_el = u_max[b_trueCount - 1];
      u_max_constrain[b_trueCount - 1] = W_act_el;
      if (!rtIsNaN(W_act_el)) {
        if (u_max[b_trueCount - 1] < 0.0) {
          u_max_constrain[b_trueCount - 1] = -1.0;
        } else {
          u_max_constrain[b_trueCount - 1] = (u_max[b_trueCount - 1] > 0.0);
        }
      }

      i_free[b_trueCount - 1] = false;
      iter++;
    }
  }

  *N_iterations = b_N_iterations;

  /* Scale back the increment to remove the normalization */
  u_out[0] *= gain_motor;
  u_out[4] *= gain_el;
  u_out[8] *= gain_az;
  u_out[1] *= gain_motor;
  u_out[5] *= gain_el;
  u_out[9] *= gain_az;
  u_out[2] *= gain_motor;
  u_out[6] *= gain_el;
  u_out[10] *= gain_az;
  u_out[3] *= gain_motor;
  u_out[7] *= gain_el;
  u_out[11] *= gain_az;
  c_compute_acc_nonlinear_earth_r(u_out, Theta, Phi, Psi, p, q, r, K_p_T, K_p_M,
    m, I_xx, I_yy, I_zz, l_1, l_2, l_3, l_4, l_z, Cl_alpha, Cd_zero, K_Cd,
    Cm_alpha, Cm_zero, rho, V, S, wing_chord, flight_path_angle, Beta, v);
  for (i = 0; i < 6; i++) {
    residuals[i] -= v[i];
  }

  *elapsed_time = toc();
  *N_evaluation = b_N_iterations;
  for (i = 0; i < 12; i++) {
    i_free[i] = rtIsNaN(u_out[i]);
  }

  aoffset = i_free[0];
  for (k = 0; k < 11; k++) {
    aoffset += i_free[k + 1];
  }

  if (aoffset > 0.5) {
    *exitflag = -1.0;
  } else {
    *exitflag = 1.0;
  }
}

/*
 * Arguments    : const double u_in[12]
 *                double Theta
 *                double Phi
 *                double Psi
 *                double p
 *                double q
 *                double r
 *                double K_p_T
 *                double K_p_M
 *                double m
 *                double I_xx
 *                double I_yy
 *                double I_zz
 *                double l_1
 *                double l_2
 *                double l_3
 *                double l_4
 *                double l_z
 *                double Cl_alpha
 *                double Cd_zero
 *                double K_Cd
 *                double Cm_alpha
 *                double Cm_zero
 *                double rho
 *                double V
 *                double S
 *                double wing_chord
 *                double flight_path_angle
 *                double Beta
 *                double computed_acc[6]
 * Return Type  : void
 */
static void c_compute_acc_nonlinear_earth_r(const double u_in[12], double Theta,
  double Phi, double Psi, double p, double q, double r, double K_p_T, double
  K_p_M, double m, double I_xx, double I_yy, double I_zz, double l_1, double l_2,
  double l_3, double l_4, double l_z, double Cl_alpha, double Cd_zero, double
  K_Cd, double Cm_alpha, double Cm_zero, double rho, double V, double S, double
  wing_chord, double flight_path_angle, double Beta, double computed_acc[6])
{
  double ab_computed_acc_tmp;
  double b_computed_acc_tmp;
  double b_computed_acc_tmp_tmp;
  double bb_computed_acc_tmp;
  double c_computed_acc_tmp;
  double c_computed_acc_tmp_tmp;
  double cb_computed_acc_tmp;
  double computed_acc_tmp;
  double computed_acc_tmp_tmp;
  double d_computed_acc_tmp;
  double d_computed_acc_tmp_tmp;
  double db_computed_acc_tmp;
  double e_computed_acc_tmp;
  double eb_computed_acc_tmp;
  double f_computed_acc_tmp;
  double fb_computed_acc_tmp;
  double g_computed_acc_tmp;
  double gb_computed_acc_tmp;
  double h_computed_acc_tmp;
  double hb_computed_acc_tmp;
  double i_computed_acc_tmp;
  double j_computed_acc_tmp;
  double k_computed_acc_tmp;
  double l_computed_acc_tmp;
  double m_computed_acc_tmp;
  double n_computed_acc_tmp;
  double o_computed_acc_tmp;
  double p_computed_acc_tmp;
  double q_computed_acc_tmp;
  double r_computed_acc_tmp;
  double s_computed_acc_tmp;
  double t_computed_acc_tmp;
  double u_computed_acc_tmp;
  double v_computed_acc_tmp;
  double w_computed_acc_tmp;
  double x_computed_acc_tmp;
  double y_computed_acc_tmp;

  /*      computed_acc(1) = -((sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta))*(K_p_T*cos(b_1)*cos(g_1)*Omega_1^2 + K_p_T*cos(b_2)*cos(g_2)*Omega_2^2 + K_p_T*cos(b_3)*cos(g_3)*Omega_3^2 + K_p_T*cos(b_4)*cos(g_4)*Omega_4^2) + (cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta))*(K_p_T*cos(b_1)*sin(g_1)*Omega_1^2 + K_p_T*cos(b_2)*sin(g_2)*Omega_2^2 + K_p_T*cos(b_3)*sin(g_3)*Omega_3^2 + K_p_T*cos(b_4)*sin(g_4)*Omega_4^2) + cos(Psi)*cos(Theta)*(K_p_T*sin(b_1)*Omega_1^2 + K_p_T*sin(b_2)*Omega_2^2 + K_p_T*sin(b_3)*Omega_3^2 + K_p_T*sin(b_4)*Omega_4^2) + (S*V^2*rho*(Cd_zero + Cl_alpha^2*K_Cd*(Theta - flight_path_angle)^2)*(cos(Beta)*sin(Theta - flight_path_angle)*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta)) - sin(Beta)*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta)) + cos(Beta)*cos(Psi)*cos(Theta)*cos(Theta - flight_path_angle)))/2 + (Cl_alpha*S*V^2*rho*(cos(Theta - flight_path_angle)*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta)) - cos(Psi)*cos(Theta)*sin(Theta - flight_path_angle))*(Theta - flight_path_angle))/2)/m; */
  /*      computed_acc(2) = ((cos(Psi)*sin(Phi) - cos(Phi)*sin(Psi)*sin(Theta))*(K_p_T*cos(b_1)*cos(g_1)*Omega_1^2 + K_p_T*cos(b_2)*cos(g_2)*Omega_2^2 + K_p_T*cos(b_3)*cos(g_3)*Omega_3^2 + K_p_T*cos(b_4)*cos(g_4)*Omega_4^2) + (cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta))*(K_p_T*cos(b_1)*sin(g_1)*Omega_1^2 + K_p_T*cos(b_2)*sin(g_2)*Omega_2^2 + K_p_T*cos(b_3)*sin(g_3)*Omega_3^2 + K_p_T*cos(b_4)*sin(g_4)*Omega_4^2) - cos(Theta)*sin(Psi)*(K_p_T*sin(b_1)*Omega_1^2 + K_p_T*sin(b_2)*Omega_2^2 + K_p_T*sin(b_3)*Omega_3^2 + K_p_T*sin(b_4)*Omega_4^2) - (S*V^2*rho*(Cd_zero + Cl_alpha^2*K_Cd*(Theta - flight_path_angle)^2)*(sin(Beta)*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta)) - cos(Beta)*sin(Theta - flight_path_angle)*(cos(Psi)*sin(Phi) - cos(Phi)*sin(Psi)*sin(Theta)) + cos(Beta)*cos(Theta)*sin(Psi)*cos(Theta - flight_path_angle)))/2 + (Cl_alpha*S*V^2*rho*(cos(Theta - flight_path_angle)*(cos(Psi)*sin(Phi) - cos(Phi)*sin(Psi)*sin(Theta)) + cos(Theta)*sin(Psi)*sin(Theta - flight_path_angle))*(Theta - flight_path_angle))/2)/m; */
  /*      computed_acc(3) =                                                 981/100 - (cos(Phi)*cos(Theta)*(K_p_T*cos(b_1)*cos(g_1)*Omega_1^2 + K_p_T*cos(b_2)*cos(g_2)*Omega_2^2 + K_p_T*cos(b_3)*cos(g_3)*Omega_3^2 + K_p_T*cos(b_4)*cos(g_4)*Omega_4^2) - cos(Theta)*sin(Phi)*(K_p_T*cos(b_1)*sin(g_1)*Omega_1^2 + K_p_T*cos(b_2)*sin(g_2)*Omega_2^2 + K_p_T*cos(b_3)*sin(g_3)*Omega_3^2 + K_p_T*cos(b_4)*sin(g_4)*Omega_4^2) - (cos(Psi)*sin(Phi) - cos(Phi)*sin(Psi)*sin(Theta))*(K_p_T*sin(b_1)*Omega_1^2 + K_p_T*sin(b_2)*Omega_2^2 + K_p_T*sin(b_3)*Omega_3^2 + K_p_T*sin(b_4)*Omega_4^2) + (S*V^2*rho*(Cd_zero + Cl_alpha^2*K_Cd*(Theta - flight_path_angle)^2)*(sin(Beta)*cos(Theta)*sin(Phi) - cos(Beta)*cos(Theta - flight_path_angle)*(cos(Psi)*sin(Phi) - cos(Phi)*sin(Psi)*sin(Theta)) + cos(Beta)*cos(Phi)*cos(Theta)*sin(Theta - flight_path_angle)))/2 + (Cl_alpha*S*V^2*rho*(sin(Theta - flight_path_angle)*(cos(Psi)*sin(Phi) - cos(Phi)*sin(Psi)*sin(Theta)) + cos(Phi)*cos(Theta)*cos(Theta - flight_path_angle))*(Theta - flight_path_angle))/2)/m; */
  computed_acc_tmp = cos(Phi);
  b_computed_acc_tmp = sin(Psi);
  c_computed_acc_tmp = cos(Psi);
  d_computed_acc_tmp = sin(Phi);
  computed_acc_tmp_tmp = sin(Theta);
  b_computed_acc_tmp_tmp = Theta - flight_path_angle;
  e_computed_acc_tmp = cos(b_computed_acc_tmp_tmp);
  f_computed_acc_tmp = sin(b_computed_acc_tmp_tmp);
  g_computed_acc_tmp = cos(Theta);
  h_computed_acc_tmp = V;
  if (!rtIsNaN(V)) {
    if (V < 0.0) {
      h_computed_acc_tmp = -1.0;
    } else {
      h_computed_acc_tmp = (V > 0.0);
    }
  }

  i_computed_acc_tmp = sin(u_in[4]);
  j_computed_acc_tmp = sin(u_in[5]);
  k_computed_acc_tmp = sin(u_in[6]);
  l_computed_acc_tmp = sin(u_in[7]);
  m_computed_acc_tmp = cos(u_in[4]);
  n_computed_acc_tmp = cos(u_in[8]);
  o_computed_acc_tmp = cos(u_in[5]);
  p_computed_acc_tmp = cos(u_in[9]);
  q_computed_acc_tmp = cos(u_in[6]);
  r_computed_acc_tmp = cos(u_in[10]);
  s_computed_acc_tmp = cos(u_in[7]);
  t_computed_acc_tmp = cos(u_in[11]);
  u_computed_acc_tmp = sin(u_in[8]);
  v_computed_acc_tmp = sin(u_in[9]);
  w_computed_acc_tmp = sin(u_in[10]);
  x_computed_acc_tmp = sin(u_in[11]);
  y_computed_acc_tmp = V * V;
  ab_computed_acc_tmp = u_in[0] * u_in[0];
  bb_computed_acc_tmp = u_in[1] * u_in[1];
  cb_computed_acc_tmp = u_in[2] * u_in[2];
  db_computed_acc_tmp = u_in[3] * u_in[3];
  c_computed_acc_tmp_tmp = S * y_computed_acc_tmp * rho;
  eb_computed_acc_tmp = c_computed_acc_tmp_tmp * cos(Beta);
  y_computed_acc_tmp = Cl_alpha * S * y_computed_acc_tmp * rho;
  fb_computed_acc_tmp = c_computed_acc_tmp_tmp * sin(Beta);
  gb_computed_acc_tmp = Cd_zero + Cl_alpha * Cl_alpha * K_Cd *
    (b_computed_acc_tmp_tmp * b_computed_acc_tmp_tmp);
  hb_computed_acc_tmp = eb_computed_acc_tmp * f_computed_acc_tmp *
    gb_computed_acc_tmp / 2.0 + y_computed_acc_tmp * e_computed_acc_tmp *
    b_computed_acc_tmp_tmp / 2.0;
  e_computed_acc_tmp = eb_computed_acc_tmp * e_computed_acc_tmp *
    gb_computed_acc_tmp / 2.0 - y_computed_acc_tmp * f_computed_acc_tmp *
    b_computed_acc_tmp_tmp / 2.0;
  f_computed_acc_tmp = fb_computed_acc_tmp * gb_computed_acc_tmp;
  computed_acc[0] = -((((((d_computed_acc_tmp * b_computed_acc_tmp +
    computed_acc_tmp * c_computed_acc_tmp * computed_acc_tmp_tmp) * (((K_p_T
    * m_computed_acc_tmp * n_computed_acc_tmp * ab_computed_acc_tmp + K_p_T
    * o_computed_acc_tmp * p_computed_acc_tmp * bb_computed_acc_tmp) + K_p_T
    * q_computed_acc_tmp * r_computed_acc_tmp * cb_computed_acc_tmp) + K_p_T
    * s_computed_acc_tmp * t_computed_acc_tmp * db_computed_acc_tmp) +
    hb_computed_acc_tmp * (sin(Phi) * sin(Psi) + cos(Phi) * cos(Psi) * sin(Theta)))
    + (computed_acc_tmp * b_computed_acc_tmp - c_computed_acc_tmp *
       d_computed_acc_tmp * computed_acc_tmp_tmp) * (((K_p_T * cos(u_in[4]) *
    u_computed_acc_tmp * ab_computed_acc_tmp + K_p_T * cos(u_in[5]) *
    v_computed_acc_tmp * bb_computed_acc_tmp) + K_p_T * cos(u_in[6]) *
    w_computed_acc_tmp * cb_computed_acc_tmp) + K_p_T * cos(u_in[7]) *
    x_computed_acc_tmp * db_computed_acc_tmp)) + c_computed_acc_tmp *
                        g_computed_acc_tmp * (((K_p_T * i_computed_acc_tmp *
    ab_computed_acc_tmp + K_p_T * j_computed_acc_tmp * bb_computed_acc_tmp) +
    K_p_T * k_computed_acc_tmp * cb_computed_acc_tmp) + K_p_T
    * l_computed_acc_tmp * db_computed_acc_tmp)) + cos(Psi) * cos(Theta) *
                       h_computed_acc_tmp * e_computed_acc_tmp) -
                      f_computed_acc_tmp * (cos(Phi) * sin(Psi) - cos(Psi) * sin
    (Phi) * sin(Theta)) / 2.0) / m;
  d_computed_acc_tmp_tmp = cos(Psi) * sin(Phi) - cos(Phi) * sin(Psi) *
    computed_acc_tmp_tmp;
  c_computed_acc_tmp = ((K_p_T * sin(u_in[4]) * (u_in[0] * u_in[0]) + K_p_T
    * sin(u_in[5]) * (u_in[1] * u_in[1])) + K_p_T * sin(u_in[6]) * (u_in[2] *
    u_in[2])) + K_p_T * sin(u_in[7]) * (u_in[3] * u_in[3]);
  y_computed_acc_tmp = ((K_p_T * cos(u_in[4]) * sin(u_in[8]) * (u_in[0] * u_in[0])
    + K_p_T * cos(u_in[5]) * sin(u_in[9]) * (u_in[1] * u_in[1])) + K_p_T * cos
                        (u_in[6]) * sin(u_in[10]) * (u_in[2] * u_in[2])) + K_p_T
    * cos(u_in[7]) * sin(u_in[11]) * (u_in[3] * u_in[3]);
  eb_computed_acc_tmp = ((K_p_T * cos(u_in[4]) * cos(u_in[8]) * (u_in[0] * u_in
    [0]) + K_p_T * cos(u_in[5]) * cos(u_in[9]) * (u_in[1] * u_in[1])) + K_p_T
    * cos(u_in[6]) * cos(u_in[10]) * (u_in[2] * u_in[2])) + K_p_T * cos(u_in[7])
    * cos(u_in[11]) * (u_in[3] * u_in[3]);
  computed_acc[1] = (((((d_computed_acc_tmp_tmp * eb_computed_acc_tmp +
    hb_computed_acc_tmp * d_computed_acc_tmp_tmp) + (cos(Phi) * cos(Psi) + sin
    (Phi) * sin(Psi) * computed_acc_tmp_tmp) * y_computed_acc_tmp) -
                       g_computed_acc_tmp * b_computed_acc_tmp *
                       c_computed_acc_tmp) - cos(Theta) * sin(Psi) *
                      h_computed_acc_tmp * e_computed_acc_tmp) -
                     f_computed_acc_tmp * (cos(Phi) * cos(Psi) + sin(Phi) * sin
    (Psi) * sin(Theta)) / 2.0) / m;
  computed_acc[2] = (((((d_computed_acc_tmp_tmp * c_computed_acc_tmp +
    h_computed_acc_tmp * e_computed_acc_tmp * d_computed_acc_tmp_tmp) +
                        g_computed_acc_tmp * d_computed_acc_tmp *
                        y_computed_acc_tmp) - computed_acc_tmp *
                       g_computed_acc_tmp * eb_computed_acc_tmp) - cos(Phi) *
                      cos(Theta) * hb_computed_acc_tmp) - fb_computed_acc_tmp *
                     g_computed_acc_tmp * d_computed_acc_tmp *
                     gb_computed_acc_tmp / 2.0) / m + 9.81;

  /*      computed_acc(4) =                                                                            (K_p_M*Omega_1^2*sin(b_1) - K_p_M*Omega_2^2*sin(b_2) + K_p_M*Omega_3^2*sin(b_3) - K_p_M*Omega_4^2*sin(b_4) + I_yy*q*r - I_zz*q*r + K_p_T*Omega_1^2*l_1*cos(b_1)*cos(g_1) - K_p_T*Omega_2^2*l_1*cos(b_2)*cos(g_2) - K_p_T*Omega_3^2*l_2*cos(b_3)*cos(g_3) + K_p_T*Omega_4^2*l_2*cos(b_4)*cos(g_4) + K_p_T*Omega_1^2*l_z*cos(b_1)*sin(g_1) + K_p_T*Omega_2^2*l_z*cos(b_2)*sin(g_2) + K_p_T*Omega_3^2*l_z*cos(b_3)*sin(g_3) + K_p_T*Omega_4^2*l_z*cos(b_4)*sin(g_4))/I_xx; */
  /*      computed_acc(5) = (I_zz*p*r - I_xx*p*r + K_p_T*Omega_1^2*l_z*sin(b_1) + K_p_T*Omega_2^2*l_z*sin(b_2) + K_p_T*Omega_3^2*l_z*sin(b_3) + K_p_T*Omega_4^2*l_z*sin(b_4) - K_p_M*Omega_1^2*cos(b_1)*sin(g_1) + K_p_M*Omega_2^2*cos(b_2)*sin(g_2) - K_p_M*Omega_3^2*cos(b_3)*sin(g_3) + K_p_M*Omega_4^2*cos(b_4)*sin(g_4) + (S*V^2*rho*wing_chord*(Cm_zero + Cm_alpha*(Theta - flight_path_angle)))/2 + K_p_T*Omega_1^2*l_4*cos(b_1)*cos(g_1) + K_p_T*Omega_2^2*l_4*cos(b_2)*cos(g_2) - K_p_T*Omega_3^2*l_3*cos(b_3)*cos(g_3) - K_p_T*Omega_4^2*l_3*cos(b_4)*cos(g_4))/I_yy; */
  /*      computed_acc(6) =                                                                        (I_xx*p*q - I_yy*p*q - K_p_T*Omega_1^2*l_1*sin(b_1) + K_p_T*Omega_2^2*l_1*sin(b_2) + K_p_T*Omega_3^2*l_2*sin(b_3) - K_p_T*Omega_4^2*l_2*sin(b_4) + K_p_M*Omega_1^2*cos(b_1)*cos(g_1) - K_p_M*Omega_2^2*cos(b_2)*cos(g_2) + K_p_M*Omega_3^2*cos(b_3)*cos(g_3) - K_p_M*Omega_4^2*cos(b_4)*cos(g_4) + K_p_T*Omega_1^2*l_4*cos(b_1)*sin(g_1) + K_p_T*Omega_2^2*l_4*cos(b_2)*sin(g_2) - K_p_T*Omega_3^2*l_3*cos(b_3)*sin(g_3) - K_p_T*Omega_4^2*l_3*cos(b_4)*sin(g_4))/I_zz; */
  computed_acc_tmp = K_p_T * ab_computed_acc_tmp;
  b_computed_acc_tmp = K_p_T * bb_computed_acc_tmp;
  c_computed_acc_tmp = K_p_T * cb_computed_acc_tmp;
  d_computed_acc_tmp = K_p_T * db_computed_acc_tmp;
  e_computed_acc_tmp = computed_acc_tmp * l_z;
  f_computed_acc_tmp = b_computed_acc_tmp * l_z;
  g_computed_acc_tmp = c_computed_acc_tmp * l_z;
  h_computed_acc_tmp = d_computed_acc_tmp * l_z;
  y_computed_acc_tmp = K_p_M * ab_computed_acc_tmp;
  ab_computed_acc_tmp = K_p_M * bb_computed_acc_tmp;
  bb_computed_acc_tmp = K_p_M * cb_computed_acc_tmp;
  cb_computed_acc_tmp = K_p_M * db_computed_acc_tmp;
  db_computed_acc_tmp = computed_acc_tmp * l_1;
  eb_computed_acc_tmp = b_computed_acc_tmp * l_1;
  fb_computed_acc_tmp = c_computed_acc_tmp * l_2;
  gb_computed_acc_tmp = d_computed_acc_tmp * l_2;
  computed_acc[3] = (((((((((((((y_computed_acc_tmp * i_computed_acc_tmp -
    ab_computed_acc_tmp * j_computed_acc_tmp) + bb_computed_acc_tmp *
    k_computed_acc_tmp) - cb_computed_acc_tmp * l_computed_acc_tmp) + I_yy * q *
    r) - I_zz * q * r) + db_computed_acc_tmp * m_computed_acc_tmp *
    n_computed_acc_tmp) - eb_computed_acc_tmp * o_computed_acc_tmp *
    p_computed_acc_tmp) - fb_computed_acc_tmp * q_computed_acc_tmp *
    r_computed_acc_tmp) + gb_computed_acc_tmp * s_computed_acc_tmp *
    t_computed_acc_tmp) + e_computed_acc_tmp * m_computed_acc_tmp *
                        u_computed_acc_tmp) + f_computed_acc_tmp *
                       o_computed_acc_tmp * v_computed_acc_tmp) +
                      g_computed_acc_tmp * q_computed_acc_tmp *
                      w_computed_acc_tmp) + h_computed_acc_tmp *
                     s_computed_acc_tmp * x_computed_acc_tmp) / I_xx;
  hb_computed_acc_tmp = I_xx * p;
  y_computed_acc_tmp *= m_computed_acc_tmp;
  ab_computed_acc_tmp *= o_computed_acc_tmp;
  bb_computed_acc_tmp *= q_computed_acc_tmp;
  cb_computed_acc_tmp *= s_computed_acc_tmp;
  computed_acc_tmp = computed_acc_tmp * l_4 * m_computed_acc_tmp;
  b_computed_acc_tmp = b_computed_acc_tmp * l_4 * o_computed_acc_tmp;
  c_computed_acc_tmp = c_computed_acc_tmp * l_3 * q_computed_acc_tmp;
  d_computed_acc_tmp = d_computed_acc_tmp * l_3 * s_computed_acc_tmp;
  computed_acc[4] = ((((((((((((((I_zz * p * r - hb_computed_acc_tmp * r) +
    e_computed_acc_tmp * i_computed_acc_tmp) + f_computed_acc_tmp *
    j_computed_acc_tmp) + g_computed_acc_tmp * k_computed_acc_tmp) +
    h_computed_acc_tmp * l_computed_acc_tmp) - y_computed_acc_tmp *
    u_computed_acc_tmp) + ab_computed_acc_tmp * v_computed_acc_tmp) -
    bb_computed_acc_tmp * w_computed_acc_tmp) + cb_computed_acc_tmp *
    x_computed_acc_tmp) + c_computed_acc_tmp_tmp * wing_chord * (Cm_zero +
    Cm_alpha * b_computed_acc_tmp_tmp) / 2.0) + computed_acc_tmp *
                        n_computed_acc_tmp) + b_computed_acc_tmp *
                       p_computed_acc_tmp) - c_computed_acc_tmp *
                      r_computed_acc_tmp) - d_computed_acc_tmp *
                     t_computed_acc_tmp) / I_yy;
  computed_acc[5] = (((((((((((((hb_computed_acc_tmp * q - I_yy * p * q) -
    db_computed_acc_tmp * i_computed_acc_tmp) + eb_computed_acc_tmp *
    j_computed_acc_tmp) + fb_computed_acc_tmp * k_computed_acc_tmp) -
    gb_computed_acc_tmp * l_computed_acc_tmp) + y_computed_acc_tmp *
    n_computed_acc_tmp) - ab_computed_acc_tmp * p_computed_acc_tmp) +
    bb_computed_acc_tmp * r_computed_acc_tmp) - cb_computed_acc_tmp *
    t_computed_acc_tmp) + computed_acc_tmp * u_computed_acc_tmp) +
                       b_computed_acc_tmp * v_computed_acc_tmp) -
                      c_computed_acc_tmp * w_computed_acc_tmp) -
                     d_computed_acc_tmp * x_computed_acc_tmp) / I_zz;
}

/*
 * cost = gamma_quadratic*(W_act_motor^2*(u_in(1) - desired_motor_value/gain_motor)^2 + W_act_motor^2*(u_in(2) - desired_motor_value/gain_motor)^2 + W_act_motor^2*(u_in(3) - desired_motor_value/gain_motor)^2 + W_act_motor^2*(u_in(4) - desired_motor_value/gain_motor)^2 + W_act_tilt_el^2*(u_in(5) - desired_el_value/gain_el)^2 + W_act_tilt_el^2*(u_in(6) - desired_el_value/gain_el)^2 + W_act_tilt_el^2*(u_in(7) - desired_el_value/gain_el)^2 + W_act_tilt_el^2*(u_in(8) - desired_el_value/gain_el)^2 + W_act_tilt_az^2*(u_in(9) - desired_az_value/gain_az)^2 + W_act_tilt_az^2*(u_in(10) - desired_az_value/gain_az)^2 + W_act_tilt_az^2*(u_in(11) - desired_az_value/gain_az)^2 + W_act_tilt_az^2*(u_in(12) - desired_az_value/gain_az)^2) + W_dv_1^2*(dv_global(1) + (K_p_T*gain_motor^2*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta))*(cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)*u_in(4)^2) + K_p_T*gain_motor^2*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta))*(cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*u_in(4)^2) + K_p_T*gain_motor^2*cos(Psi)*cos(Theta)*(sin(u_in(5)*gain_el)*u_in(1)^2 + sin(u_in(6)*gain_el)*u_in(2)^2 + sin(u_in(7)*gain_el)*u_in(3)^2 + sin(u_in(8)*gain_el)*u_in(4)^2))/m)^2 + (W_dv_3^2*((100*(K_p_T*gain_motor^2*(cos(Psi)*sin(Phi) - cos(Phi)*sin(Psi)*sin(Theta))*(sin(u_in(5)*gain_el)*u_in(1)^2 + sin(u_in(6)*gain_el)*u_in(2)^2 + sin(u_in(7)*gain_el)*u_in(3)^2 + sin(u_in(8)*gain_el)*u_in(4)^2) + K_p_T*gain_motor^2*cos(Theta)*sin(Phi)*(cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*u_in(4)^2) - K_p_T*gain_motor^2*cos(Phi)*cos(Theta)*(cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)*u_in(4)^2)))/m - 100*dv_global(3) + 981)^2)/10000 + W_dv_2^2*(dv_global(2) - (K_p_T*gain_motor^2*(cos(Psi)*sin(Phi) - cos(Phi)*sin(Psi)*sin(Theta))*(cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)*u_in(4)^2) + K_p_T*gain_motor^2*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta))*(cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*u_in(4)^2) - K_p_T*gain_motor^2*cos(Theta)*sin(Psi)*(sin(u_in(5)*gain_el)*u_in(1)^2 + sin(u_in(6)*gain_el)*u_in(2)^2 + sin(u_in(7)*gain_el)*u_in(3)^2 + sin(u_in(8)*gain_el)*u_in(4)^2))/m)^2 + (W_dv_6^2*(I_zz*dv_global(6) - I_xx*p*q + I_yy*p*q + K_p_T*u_in(1)^2*gain_motor^2*l_1*sin(u_in(5)*gain_el) - K_p_T*u_in(2)^2*gain_motor^2*l_1*sin(u_in(6)*gain_el) - K_p_T*u_in(3)^2*gain_motor^2*l_2*sin(u_in(7)*gain_el) + K_p_T*u_in(4)^2*gain_motor^2*l_2*sin(u_in(8)*gain_el) - K_p_M*u_in(1)^2*gain_motor^2*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az) + K_p_M*u_in(2)^2*gain_motor^2*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az) - K_p_M*u_in(3)^2*gain_motor^2*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az) + K_p_M*u_in(4)^2*gain_motor^2*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az) - K_p_T*u_in(1)^2*gain_motor^2*l_4*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az) - K_p_T*u_in(2)^2*gain_motor^2*l_4*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az) + K_p_T*u_in(3)^2*gain_motor^2*l_3*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az) + K_p_T*u_in(4)^2*gain_motor^2*l_3*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az))^2)/I_zz^2 + (W_dv_4^2*(I_yy*q*r - I_xx*dv_global(4) - I_zz*q*r + K_p_M*u_in(1)^2*gain_motor^2*sin(u_in(5)*gain_el) - K_p_M*u_in(2)^2*gain_motor^2*sin(u_in(6)*gain_el) + K_p_M*u_in(3)^2*gain_motor^2*sin(u_in(7)*gain_el) - K_p_M*u_in(4)^2*gain_motor^2*sin(u_in(8)*gain_el) + K_p_T*u_in(1)^2*gain_motor^2*l_1*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az) - K_p_T*u_in(2)^2*gain_motor^2*l_1*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az) - K_p_T*u_in(3)^2*gain_motor^2*l_2*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az) + K_p_T*u_in(4)^2*gain_motor^2*l_2*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az) + K_p_T*u_in(1)^2*gain_motor^2*l_z*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az) + K_p_T*u_in(2)^2*gain_motor^2*l_z*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az) + K_p_T*u_in(3)^2*gain_motor^2*l_z*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az) + K_p_T*u_in(4)^2*gain_motor^2*l_z*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az))^2)/I_xx^2 + (W_dv_5^2*(2*I_zz*p*r - 2*I_xx*p*r - 2*I_yy*dv_global(5) + 2*K_p_T*u_in(1)^2*gain_motor^2*l_z*sin(u_in(5)*gain_el) + 2*K_p_T*u_in(2)^2*gain_motor^2*l_z*sin(u_in(6)*gain_el) + 2*K_p_T*u_in(3)^2*gain_motor^2*l_z*sin(u_in(7)*gain_el) + 2*K_p_T*u_in(4)^2*gain_motor^2*l_z*sin(u_in(8)*gain_el) - 2*K_p_M*u_in(1)^2*gain_motor^2*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az) + 2*K_p_M*u_in(2)^2*gain_motor^2*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az) - 2*K_p_M*u_in(3)^2*gain_motor^2*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az) + 2*K_p_M*u_in(4)^2*gain_motor^2*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az) + Cm_zero*S*V^2*rho*wing_chord + 2*K_p_T*u_in(1)^2*gain_motor^2*l_4*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az) + 2*K_p_T*u_in(2)^2*gain_motor^2*l_4*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az) - 2*K_p_T*u_in(3)^2*gain_motor^2*l_3*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az) - 2*K_p_T*u_in(4)^2*gain_motor^2*l_3*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az) + Cm_alpha*S*Theta*V^2*rho*wing_chord - Cm_alpha*S*V^2*flight_path_angle*rho*wing_chord)^2)/(4*I_yy^2);
 *
 * Arguments    : const captured_var *W_act_motor
 *                const captured_var *gamma_quadratic_du
 *                const captured_var *desired_motor_value
 *                const captured_var *gain_motor
 *                const captured_var *W_dv_2
 *                const captured_var *gamma_quadratic_dv
 *                const b_captured_var *dv_global
 *                const captured_var *S
 *                const captured_var *V
 *                const captured_var *rho
 *                const captured_var *Beta
 *                const captured_var *Theta
 *                const captured_var *flight_path_angle
 *                const captured_var *K_Cd
 *                const captured_var *Cl_alpha
 *                const captured_var *Cd_zero
 *                const captured_var *Psi
 *                const captured_var *Phi
 *                const captured_var *K_p_T
 *                const captured_var *gain_el
 *                const captured_var *gain_az
 *                const captured_var *m
 *                const captured_var *W_act_tilt_el
 *                const captured_var *desired_el_value
 *                const captured_var *W_act_tilt_az
 *                const captured_var *desired_az_value
 *                const captured_var *W_dv_3
 *                const captured_var *W_dv_1
 *                const captured_var *W_dv_5
 *                const captured_var *I_zz
 *                const captured_var *p
 *                const captured_var *r
 *                const captured_var *I_xx
 *                const captured_var *I_yy
 *                const captured_var *l_z
 *                const captured_var *K_p_M
 *                const captured_var *Cm_zero
 *                const captured_var *wing_chord
 *                const captured_var *l_4
 *                const captured_var *l_3
 *                const captured_var *Cm_alpha
 *                const captured_var *W_dv_6
 *                const captured_var *q
 *                const captured_var *l_1
 *                const captured_var *l_2
 *                const captured_var *W_dv_4
 *                const double u_in[12]
 *                double *cost
 *                double computed_gradient[12]
 * Return Type  : void
 */
static void c_compute_cost_and_gradient_fcn(const captured_var *W_act_motor,
  const captured_var *gamma_quadratic_du, const captured_var
  *desired_motor_value, const captured_var *gain_motor, const captured_var
  *W_dv_2, const captured_var *gamma_quadratic_dv, const b_captured_var
  *dv_global, const captured_var *S, const captured_var *V, const captured_var
  *rho, const captured_var *Beta, const captured_var *Theta, const captured_var *
  flight_path_angle, const captured_var *K_Cd, const captured_var *Cl_alpha,
  const captured_var *Cd_zero, const captured_var *Psi, const captured_var *Phi,
  const captured_var *K_p_T, const captured_var *gain_el, const captured_var
  *gain_az, const captured_var *m, const captured_var *W_act_tilt_el, const
  captured_var *desired_el_value, const captured_var *W_act_tilt_az, const
  captured_var *desired_az_value, const captured_var *W_dv_3, const captured_var
  *W_dv_1, const captured_var *W_dv_5, const captured_var *I_zz, const
  captured_var *p, const captured_var *r, const captured_var *I_xx, const
  captured_var *I_yy, const captured_var *l_z, const captured_var *K_p_M, const
  captured_var *Cm_zero, const captured_var *wing_chord, const captured_var *l_4,
  const captured_var *l_3, const captured_var *Cm_alpha, const captured_var
  *W_dv_6, const captured_var *q, const captured_var *l_1, const captured_var
  *l_2, const captured_var *W_dv_4, const double u_in[12], double *cost, double
  computed_gradient[12])
{
  double a;
  double a_tmp;
  double a_tmp_tmp;
  double ab_a;
  double ab_a_tmp;
  double ab_a_tmp_tmp;
  double ac_a_tmp;
  double b_a;
  double b_a_tmp;
  double b_a_tmp_tmp;
  double b_cost_tmp;
  double b_cost_tmp_tmp;
  double bb_a;
  double bb_a_tmp;
  double bb_a_tmp_tmp;
  double bc_a_tmp;
  double c_a;
  double c_a_tmp;
  double c_a_tmp_tmp;
  double c_cost_tmp;
  double cb_a_tmp;
  double cb_a_tmp_tmp;
  double cc_a_tmp;
  double cost_tmp;
  double cost_tmp_tmp;
  double d_a;
  double d_a_tmp;
  double d_a_tmp_tmp;
  double d_cost_tmp;
  double db_a_tmp;
  double db_a_tmp_tmp;
  double dc_a_tmp;
  double e_a;
  double e_a_tmp;
  double e_a_tmp_tmp;
  double eb_a_tmp;
  double eb_a_tmp_tmp;
  double ec_a_tmp;
  double f_a;
  double f_a_tmp;
  double f_a_tmp_tmp;
  double fb_a_tmp;
  double fb_a_tmp_tmp;
  double fc_a_tmp;
  double g_a;
  double g_a_tmp;
  double g_a_tmp_tmp;
  double gb_a_tmp;
  double gb_a_tmp_tmp;
  double gc_a_tmp;
  double h_a;
  double h_a_tmp;
  double h_a_tmp_tmp;
  double hb_a_tmp;
  double hb_a_tmp_tmp;
  double hc_a_tmp;
  double i_a;
  double i_a_tmp;
  double i_a_tmp_tmp;
  double ib_a_tmp;
  double ib_a_tmp_tmp;
  double ic_a_tmp;
  double j_a;
  double j_a_tmp;
  double j_a_tmp_tmp;
  double jb_a_tmp;
  double jc_a_tmp;
  double k_a;
  double k_a_tmp;
  double k_a_tmp_tmp;
  double kb_a_tmp;
  double kc_a_tmp;
  double l_a;
  double l_a_tmp;
  double l_a_tmp_tmp;
  double lb_a_tmp;
  double lc_a_tmp;
  double m_a;
  double m_a_tmp;
  double m_a_tmp_tmp;
  double mb_a_tmp;
  double mc_a_tmp;
  double n_a;
  double n_a_tmp;
  double n_a_tmp_tmp;
  double nb_a_tmp;
  double o_a;
  double o_a_tmp;
  double o_a_tmp_tmp;
  double ob_a_tmp;
  double p_a;
  double p_a_tmp;
  double p_a_tmp_tmp;
  double pb_a_tmp;
  double q_a;
  double q_a_tmp;
  double q_a_tmp_tmp;
  double qb_a_tmp;
  double r_a;
  double r_a_tmp;
  double r_a_tmp_tmp;
  double rb_a_tmp;
  double s_a;
  double s_a_tmp;
  double s_a_tmp_tmp;
  double sb_a_tmp;
  double sigma_3;
  double sigma_4;
  double sigma_5;
  double sigma_6;
  double sigma_7;
  double t_a;
  double t_a_tmp;
  double t_a_tmp_tmp;
  double tb_a_tmp;
  double u_a;
  double u_a_tmp;
  double u_a_tmp_tmp;
  double ub_a_tmp;
  double v_a;
  double v_a_tmp;
  double v_a_tmp_tmp;
  double vb_a_tmp;
  double w_a;
  double w_a_tmp;
  double w_a_tmp_tmp;
  double wb_a_tmp;
  double x;
  double x_a;
  double x_a_tmp;
  double x_a_tmp_tmp;
  double xb_a_tmp;
  double y_a;
  double y_a_tmp;
  double y_a_tmp_tmp;
  double yb_a_tmp;
  bool b;

  /*      %no aerodynamic on forces: */
  /*      cost = W_dv_1^2*gamma_quadratic_dv*(dv_global(1) + (K_p_T*gain_motor^2*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta))*(cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)*u_in(4)^2) + K_p_T*gain_motor^2*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta))*(cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*u_in(4)^2) + K_p_T*gain_motor^2*cos(Psi)*cos(Theta)*(sin(u_in(5)*gain_el)*u_in(1)^2 + sin(u_in(6)*gain_el)*u_in(2)^2 + sin(u_in(7)*gain_el)*u_in(3)^2 + sin(u_in(8)*gain_el)*u_in(4)^2))/m)^2 + (W_dv_3^2*gamma_quadratic_dv*((100*(K_p_T*gain_motor^2*(cos(Psi)*sin(Phi) - cos(Phi)*sin(Psi)*sin(Theta))*(sin(u_in(5)*gain_el)*u_in(1)^2 + sin(u_in(6)*gain_el)*u_in(2)^2 + sin(u_in(7)*gain_el)*u_in(3)^2 + sin(u_in(8)*gain_el)*u_in(4)^2) + K_p_T*gain_motor^2*cos(Theta)*sin(Phi)*(cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*u_in(4)^2) - K_p_T*gain_motor^2*cos(Phi)*cos(Theta)*(cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)*u_in(4)^2)))/m - 100*dv_global(3) + 981)^2)/10000 + W_dv_2^2*gamma_quadratic_dv*(dv_global(2) - (K_p_T*gain_motor^2*(cos(Psi)*sin(Phi) - cos(Phi)*sin(Psi)*sin(Theta))*(cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)*u_in(4)^2) + K_p_T*gain_motor^2*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta))*(cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*u_in(4)^2) - K_p_T*gain_motor^2*cos(Theta)*sin(Psi)*(sin(u_in(5)*gain_el)*u_in(1)^2 + sin(u_in(6)*gain_el)*u_in(2)^2 + sin(u_in(7)*gain_el)*u_in(3)^2 + sin(u_in(8)*gain_el)*u_in(4)^2))/m)^2 + W_act_motor^2*gamma_quadratic_du*(u_in(1) - desired_motor_value/gain_motor)^2 + W_act_motor^2*gamma_quadratic_du*(u_in(2) - desired_motor_value/gain_motor)^2 + W_act_motor^2*gamma_quadratic_du*(u_in(3) - desired_motor_value/gain_motor)^2 + W_act_motor^2*gamma_quadratic_du*(u_in(4) - desired_motor_value/gain_motor)^2 + W_act_tilt_el^2*gamma_quadratic_du*(u_in(5) - desired_el_value/gain_el)^2 + W_act_tilt_el^2*gamma_quadratic_du*(u_in(6) - desired_el_value/gain_el)^2 + W_act_tilt_el^2*gamma_quadratic_du*(u_in(7) - desired_el_value/gain_el)^2 + W_act_tilt_el^2*gamma_quadratic_du*(u_in(8) - desired_el_value/gain_el)^2 + W_act_tilt_az^2*gamma_quadratic_du*(u_in(9) - desired_az_value/gain_az)^2 + W_act_tilt_az^2*gamma_quadratic_du*(u_in(10) - desired_az_value/gain_az)^2 + W_act_tilt_az^2*gamma_quadratic_du*(u_in(11) - desired_az_value/gain_az)^2 + W_act_tilt_az^2*gamma_quadratic_du*(u_in(12) - desired_az_value/gain_az)^2 + (W_dv_5^2*gamma_quadratic_dv*(2*I_zz*p*r - 2*I_xx*p*r - 2*I_yy*dv_global(5) + 2*K_p_T*u_in(1)^2*gain_motor^2*l_z*sin(u_in(5)*gain_el) + 2*K_p_T*u_in(2)^2*gain_motor^2*l_z*sin(u_in(6)*gain_el) + 2*K_p_T*u_in(3)^2*gain_motor^2*l_z*sin(u_in(7)*gain_el) + 2*K_p_T*u_in(4)^2*gain_motor^2*l_z*sin(u_in(8)*gain_el) - 2*K_p_M*u_in(1)^2*gain_motor^2*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az) + 2*K_p_M*u_in(2)^2*gain_motor^2*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az) - 2*K_p_M*u_in(3)^2*gain_motor^2*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az) + 2*K_p_M*u_in(4)^2*gain_motor^2*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az) + Cm_zero*S*V^2*rho*wing_chord + 2*K_p_T*u_in(1)^2*gain_motor^2*l_4*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az) + 2*K_p_T*u_in(2)^2*gain_motor^2*l_4*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az) - 2*K_p_T*u_in(3)^2*gain_motor^2*l_3*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az) - 2*K_p_T*u_in(4)^2*gain_motor^2*l_3*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az) + Cm_alpha*S*Theta*V^2*rho*wing_chord - Cm_alpha*S*V^2*flight_path_angle*rho*wing_chord)^2)/(4*I_yy^2) + (W_dv_6^2*gamma_quadratic_dv*(I_zz*dv_global(6) - I_xx*p*q + I_yy*p*q + K_p_T*u_in(1)^2*gain_motor^2*l_1*sin(u_in(5)*gain_el) - K_p_T*u_in(2)^2*gain_motor^2*l_1*sin(u_in(6)*gain_el) - K_p_T*u_in(3)^2*gain_motor^2*l_2*sin(u_in(7)*gain_el) + K_p_T*u_in(4)^2*gain_motor^2*l_2*sin(u_in(8)*gain_el) - K_p_M*u_in(1)^2*gain_motor^2*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az) + K_p_M*u_in(2)^2*gain_motor^2*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az) - K_p_M*u_in(3)^2*gain_motor^2*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az) + K_p_M*u_in(4)^2*gain_motor^2*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az) - K_p_T*u_in(1)^2*gain_motor^2*l_4*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az) - K_p_T*u_in(2)^2*gain_motor^2*l_4*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az) + K_p_T*u_in(3)^2*gain_motor^2*l_3*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az) + K_p_T*u_in(4)^2*gain_motor^2*l_3*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az))^2)/I_zz^2 + (W_dv_4^2*gamma_quadratic_dv*(I_yy*q*r - I_xx*dv_global(4) - I_zz*q*r + K_p_M*u_in(1)^2*gain_motor^2*sin(u_in(5)*gain_el) - K_p_M*u_in(2)^2*gain_motor^2*sin(u_in(6)*gain_el) + K_p_M*u_in(3)^2*gain_motor^2*sin(u_in(7)*gain_el) - K_p_M*u_in(4)^2*gain_motor^2*sin(u_in(8)*gain_el) + K_p_T*u_in(1)^2*gain_motor^2*l_1*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az) - K_p_T*u_in(2)^2*gain_motor^2*l_1*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az) - K_p_T*u_in(3)^2*gain_motor^2*l_2*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az) + K_p_T*u_in(4)^2*gain_motor^2*l_2*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az) + K_p_T*u_in(1)^2*gain_motor^2*l_z*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az) + K_p_T*u_in(2)^2*gain_motor^2*l_z*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az) + K_p_T*u_in(3)^2*gain_motor^2*l_z*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az) + K_p_T*u_in(4)^2*gain_motor^2*l_z*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az))^2)/I_xx^2; */
  a_tmp = W_act_motor->contents;
  b_a_tmp = desired_motor_value->contents / gain_motor->contents;
  a = u_in[0] - b_a_tmp;
  b_a = u_in[1] - b_a_tmp;
  c_a = u_in[2] - b_a_tmp;
  d_a = u_in[3] - b_a_tmp;
  b_a_tmp = W_dv_2->contents;
  c_a_tmp = V->contents;
  d_a_tmp = Cl_alpha->contents;
  e_a_tmp = Theta->contents;
  f_a_tmp = flight_path_angle->contents;
  e_a = gain_motor->contents;
  f_a = gain_motor->contents;
  g_a = gain_motor->contents;
  x = c_a_tmp;
  b = rtIsNaN(c_a_tmp);
  if (!b) {
    if (c_a_tmp < 0.0) {
      x = -1.0;
    } else {
      x = (c_a_tmp > 0.0);
    }
  }

  g_a_tmp = Theta->contents - flight_path_angle->contents;
  a_tmp_tmp = cos(Phi->contents);
  b_a_tmp_tmp = cos(Psi->contents);
  c_a_tmp_tmp = sin(Phi->contents);
  d_a_tmp_tmp = sin(Psi->contents);
  e_a_tmp_tmp = sin(Theta->contents);
  f_a_tmp_tmp = b_a_tmp_tmp * c_a_tmp_tmp;
  g_a_tmp_tmp = a_tmp_tmp * d_a_tmp_tmp;
  h_a_tmp = f_a_tmp_tmp - g_a_tmp_tmp * e_a_tmp_tmp;
  i_a_tmp = u_in[8] * gain_az->contents;
  h_a_tmp_tmp = u_in[4] * gain_el->contents;
  j_a_tmp = cos(h_a_tmp_tmp);
  k_a_tmp = u_in[0] * u_in[0];
  l_a_tmp = u_in[9] * gain_az->contents;
  i_a_tmp_tmp = u_in[5] * gain_el->contents;
  m_a_tmp = cos(i_a_tmp_tmp);
  n_a_tmp = u_in[1] * u_in[1];
  o_a_tmp = u_in[10] * gain_az->contents;
  j_a_tmp_tmp = u_in[6] * gain_el->contents;
  p_a_tmp = cos(j_a_tmp_tmp);
  q_a_tmp = u_in[2] * u_in[2];
  r_a_tmp = u_in[11] * gain_az->contents;
  k_a_tmp_tmp = u_in[7] * gain_el->contents;
  s_a_tmp = cos(k_a_tmp_tmp);
  t_a_tmp = u_in[3] * u_in[3];
  u_a_tmp = cos(Beta->contents);
  v_a_tmp = cos(g_a_tmp);
  w_a_tmp = 2.0 * K_Cd->contents;
  x_a_tmp = Cl_alpha->contents * S->contents;
  y_a_tmp = sin(g_a_tmp);
  ab_a_tmp = cos(Theta->contents);
  l_a_tmp_tmp = a_tmp_tmp * b_a_tmp_tmp;
  m_a_tmp_tmp = c_a_tmp_tmp * d_a_tmp_tmp;
  bb_a_tmp = l_a_tmp_tmp + m_a_tmp_tmp * e_a_tmp_tmp;
  n_a_tmp_tmp = sin(h_a_tmp_tmp);
  o_a_tmp_tmp = sin(i_a_tmp_tmp);
  p_a_tmp_tmp = sin(j_a_tmp_tmp);
  q_a_tmp_tmp = sin(k_a_tmp_tmp);
  cb_a_tmp = ((n_a_tmp_tmp * k_a_tmp + o_a_tmp_tmp * n_a_tmp) + p_a_tmp_tmp *
              q_a_tmp) + q_a_tmp_tmp * t_a_tmp;
  r_a_tmp_tmp = sin(i_a_tmp);
  s_a_tmp_tmp = sin(l_a_tmp);
  t_a_tmp_tmp = sin(o_a_tmp);
  u_a_tmp_tmp = sin(r_a_tmp);
  v_a_tmp_tmp = j_a_tmp * r_a_tmp_tmp;
  w_a_tmp_tmp = m_a_tmp * s_a_tmp_tmp;
  x_a_tmp_tmp = p_a_tmp * t_a_tmp_tmp;
  y_a_tmp_tmp = s_a_tmp * u_a_tmp_tmp;
  db_a_tmp = ((v_a_tmp_tmp * k_a_tmp + w_a_tmp_tmp * n_a_tmp) + x_a_tmp_tmp *
              q_a_tmp) + y_a_tmp_tmp * t_a_tmp;
  ab_a_tmp_tmp = cos(i_a_tmp);
  bb_a_tmp_tmp = cos(l_a_tmp);
  cb_a_tmp_tmp = cos(o_a_tmp);
  db_a_tmp_tmp = cos(r_a_tmp);
  eb_a_tmp_tmp = j_a_tmp * ab_a_tmp_tmp;
  fb_a_tmp_tmp = m_a_tmp * bb_a_tmp_tmp;
  gb_a_tmp_tmp = p_a_tmp * cb_a_tmp_tmp;
  hb_a_tmp_tmp = s_a_tmp * db_a_tmp_tmp;
  eb_a_tmp = ((eb_a_tmp_tmp * k_a_tmp + fb_a_tmp_tmp * n_a_tmp) + gb_a_tmp_tmp *
              q_a_tmp) + hb_a_tmp_tmp * t_a_tmp;
  fb_a_tmp = d_a_tmp * d_a_tmp;
  gb_a_tmp = K_Cd->contents * fb_a_tmp;
  hb_a_tmp = c_a_tmp * c_a_tmp;
  ib_a_tmp_tmp = S->contents * hb_a_tmp * rho->contents;
  ib_a_tmp = ib_a_tmp_tmp * u_a_tmp;
  fb_a_tmp = ((gb_a_tmp * (e_a_tmp * e_a_tmp) - w_a_tmp * fb_a_tmp *
               Theta->contents * flight_path_angle->contents) + gb_a_tmp *
              (f_a_tmp * f_a_tmp)) + Cd_zero->contents;
  gb_a_tmp = x_a_tmp * hb_a_tmp * rho->contents;
  jb_a_tmp = ib_a_tmp * y_a_tmp * fb_a_tmp / 2.0 + gb_a_tmp * v_a_tmp * g_a_tmp /
    2.0;
  gb_a_tmp = ib_a_tmp * v_a_tmp * fb_a_tmp / 2.0 - gb_a_tmp * y_a_tmp * g_a_tmp /
    2.0;
  ib_a_tmp = ib_a_tmp_tmp * sin(Beta->contents);
  kb_a_tmp = ab_a_tmp * d_a_tmp_tmp;
  lb_a_tmp = ib_a_tmp * bb_a_tmp * fb_a_tmp / 2.0;
  e_a = dv_global->contents[1] - (((((jb_a_tmp * h_a_tmp + K_p_T->contents *
    (e_a * e_a) * h_a_tmp * eb_a_tmp) + K_p_T->contents * (f_a * f_a) * bb_a_tmp
    * db_a_tmp) - kb_a_tmp * x * gb_a_tmp) - K_p_T->contents * (g_a * g_a) *
    ab_a_tmp * d_a_tmp_tmp * cb_a_tmp) - lb_a_tmp) / m->contents;
  mb_a_tmp = W_act_tilt_el->contents;
  nb_a_tmp = desired_el_value->contents / gain_el->contents;
  f_a = u_in[4] - nb_a_tmp;
  g_a = u_in[5] - nb_a_tmp;
  h_a = u_in[6] - nb_a_tmp;
  i_a = u_in[7] - nb_a_tmp;
  nb_a_tmp = W_act_tilt_az->contents;
  ob_a_tmp = desired_az_value->contents / gain_az->contents;
  j_a = u_in[8] - ob_a_tmp;
  k_a = u_in[9] - ob_a_tmp;
  l_a = u_in[10] - ob_a_tmp;
  m_a = u_in[11] - ob_a_tmp;
  ob_a_tmp = W_dv_3->contents;
  n_a = gain_motor->contents;
  o_a = gain_motor->contents;
  p_a = gain_motor->contents;
  x = c_a_tmp;
  if (!b) {
    if (c_a_tmp < 0.0) {
      x = -1.0;
    } else {
      x = (c_a_tmp > 0.0);
    }
  }

  sigma_3 = a_tmp_tmp * ab_a_tmp * jb_a_tmp;
  pb_a_tmp = ib_a_tmp * ab_a_tmp * c_a_tmp_tmp * fb_a_tmp / 2.0;
  n_a = (100.0 * dv_global->contents[2] + 100.0 * (((((sigma_3 - x * gb_a_tmp *
              h_a_tmp) - K_p_T->contents * (n_a * n_a) * h_a_tmp * cb_a_tmp) -
            K_p_T->contents * (o_a * o_a) * ab_a_tmp * c_a_tmp_tmp * db_a_tmp) +
           K_p_T->contents * (p_a * p_a) * a_tmp_tmp * ab_a_tmp * eb_a_tmp) +
          pb_a_tmp) / m->contents) - 981.0;
  qb_a_tmp = W_dv_1->contents;
  o_a = gain_motor->contents;
  p_a = gain_motor->contents;
  q_a = gain_motor->contents;
  x = c_a_tmp;
  if (!b) {
    if (c_a_tmp < 0.0) {
      x = -1.0;
    } else {
      x = (c_a_tmp > 0.0);
    }
  }

  rb_a_tmp = m_a_tmp_tmp + l_a_tmp_tmp * e_a_tmp_tmp;
  sb_a_tmp = g_a_tmp_tmp - f_a_tmp_tmp * e_a_tmp_tmp;
  tb_a_tmp = b_a_tmp_tmp * ab_a_tmp;
  fb_a_tmp = ib_a_tmp * sb_a_tmp * fb_a_tmp / 2.0;
  o_a = dv_global->contents[0] + (((((jb_a_tmp * rb_a_tmp + K_p_T->contents *
    (o_a * o_a) * rb_a_tmp * eb_a_tmp) + tb_a_tmp * x * gb_a_tmp) +
    K_p_T->contents * (p_a * p_a) * sb_a_tmp * db_a_tmp) + K_p_T->contents *
    (q_a * q_a) * b_a_tmp_tmp * ab_a_tmp * cb_a_tmp) - fb_a_tmp) / m->contents;
  cb_a_tmp = W_dv_5->contents;
  p_a = gain_motor->contents;
  q_a = gain_motor->contents;
  r_a = gain_motor->contents;
  x = gain_motor->contents;
  sigma_6 = gain_motor->contents;
  sigma_7 = gain_motor->contents;
  ib_a_tmp_tmp = gain_motor->contents;
  s_a = gain_motor->contents;
  t_a = gain_motor->contents;
  u_a = gain_motor->contents;
  v_a = gain_motor->contents;
  w_a = gain_motor->contents;
  ib_a_tmp = 2.0 * K_p_T->contents;
  jb_a_tmp = 2.0 * K_p_M->contents;
  ub_a_tmp = ib_a_tmp * k_a_tmp;
  vb_a_tmp = ib_a_tmp * n_a_tmp;
  wb_a_tmp = ib_a_tmp * q_a_tmp;
  xb_a_tmp = ib_a_tmp * t_a_tmp;
  f_a_tmp_tmp = Cm_alpha->contents * S->contents;
  yb_a_tmp = (2.0 * I_zz->contents * p->contents * r->contents - 2.0 *
              I_xx->contents * p->contents * r->contents) - 2.0 * I_yy->contents
    * dv_global->contents[4];
  ac_a_tmp = jb_a_tmp * k_a_tmp;
  sigma_5 = jb_a_tmp * n_a_tmp;
  sigma_4 = jb_a_tmp * q_a_tmp;
  jb_a_tmp *= t_a_tmp;
  bc_a_tmp = Cm_zero->contents * S->contents * hb_a_tmp * rho->contents *
    wing_chord->contents;
  cc_a_tmp = f_a_tmp_tmp * Theta->contents * hb_a_tmp * rho->contents *
    wing_chord->contents;
  hb_a_tmp = f_a_tmp_tmp * hb_a_tmp * flight_path_angle->contents *
    rho->contents * wing_chord->contents;
  p_a = ((((((((((((((yb_a_tmp + ub_a_tmp * (p_a * p_a) * l_z->contents *
                      n_a_tmp_tmp) + vb_a_tmp * (q_a * q_a) * l_z->contents *
                     o_a_tmp_tmp) + wb_a_tmp * (r_a * r_a) * l_z->contents *
                    p_a_tmp_tmp) + xb_a_tmp * (x * x) * l_z->contents *
                   q_a_tmp_tmp) - ac_a_tmp * (sigma_6 * sigma_6) * j_a_tmp *
                  r_a_tmp_tmp) + sigma_5 * (sigma_7 * sigma_7) * m_a_tmp *
                 s_a_tmp_tmp) - sigma_4 * (ib_a_tmp_tmp * ib_a_tmp_tmp) *
                p_a_tmp * t_a_tmp_tmp) + jb_a_tmp * (s_a * s_a) * s_a_tmp *
               u_a_tmp_tmp) + bc_a_tmp) + ub_a_tmp * (t_a * t_a) * l_4->contents
             * j_a_tmp * ab_a_tmp_tmp) + vb_a_tmp * (u_a * u_a) * l_4->contents *
            m_a_tmp * bb_a_tmp_tmp) - wb_a_tmp * (v_a * v_a) * l_3->contents *
           p_a_tmp * cb_a_tmp_tmp) - xb_a_tmp * (w_a * w_a) * l_3->contents *
          s_a_tmp * db_a_tmp_tmp) + cc_a_tmp) - hb_a_tmp;
  q_a = I_yy->contents;
  f_a_tmp_tmp = W_dv_6->contents;
  r_a = gain_motor->contents;
  x = gain_motor->contents;
  sigma_6 = gain_motor->contents;
  sigma_7 = gain_motor->contents;
  ib_a_tmp_tmp = gain_motor->contents;
  s_a = gain_motor->contents;
  t_a = gain_motor->contents;
  u_a = gain_motor->contents;
  v_a = gain_motor->contents;
  w_a = gain_motor->contents;
  x_a = gain_motor->contents;
  y_a = gain_motor->contents;
  dc_a_tmp = K_p_T->contents * k_a_tmp;
  ec_a_tmp = K_p_T->contents * n_a_tmp;
  fc_a_tmp = K_p_T->contents * q_a_tmp;
  gc_a_tmp = K_p_T->contents * t_a_tmp;
  hc_a_tmp = K_p_M->contents * k_a_tmp;
  ic_a_tmp = K_p_M->contents * n_a_tmp;
  jc_a_tmp = K_p_M->contents * q_a_tmp;
  kc_a_tmp = K_p_M->contents * t_a_tmp;
  lc_a_tmp = (I_zz->contents * dv_global->contents[5] - I_xx->contents *
              p->contents * q->contents) + I_yy->contents * p->contents *
    q->contents;
  r_a = (((((((((((lc_a_tmp + dc_a_tmp * (r_a * r_a) * l_1->contents *
                   n_a_tmp_tmp) - ec_a_tmp * (x * x) * l_1->contents *
                  o_a_tmp_tmp) - fc_a_tmp * (sigma_6 * sigma_6) * l_2->contents *
                 p_a_tmp_tmp) + gc_a_tmp * (sigma_7 * sigma_7) * l_2->contents *
                q_a_tmp_tmp) - hc_a_tmp * (ib_a_tmp_tmp * ib_a_tmp_tmp) *
               j_a_tmp * ab_a_tmp_tmp) + ic_a_tmp * (s_a * s_a) * m_a_tmp *
              bb_a_tmp_tmp) - jc_a_tmp * (t_a * t_a) * p_a_tmp * cb_a_tmp_tmp) +
            kc_a_tmp * (u_a * u_a) * s_a_tmp * db_a_tmp_tmp) - dc_a_tmp * (v_a *
            v_a) * l_4->contents * j_a_tmp * r_a_tmp_tmp) - ec_a_tmp * (w_a *
           w_a) * l_4->contents * m_a_tmp * s_a_tmp_tmp) + fc_a_tmp * (x_a * x_a)
         * l_3->contents * p_a_tmp * t_a_tmp_tmp) + gc_a_tmp * (y_a * y_a) *
    l_3->contents * s_a_tmp * u_a_tmp_tmp;
  x = I_zz->contents;
  g_a_tmp_tmp = W_dv_4->contents;
  sigma_6 = gain_motor->contents;
  sigma_7 = gain_motor->contents;
  ib_a_tmp_tmp = gain_motor->contents;
  s_a = gain_motor->contents;
  t_a = gain_motor->contents;
  u_a = gain_motor->contents;
  v_a = gain_motor->contents;
  w_a = gain_motor->contents;
  x_a = gain_motor->contents;
  y_a = gain_motor->contents;
  ab_a = gain_motor->contents;
  bb_a = gain_motor->contents;
  mc_a_tmp = (I_yy->contents * q->contents * r->contents - I_xx->contents *
              dv_global->contents[3]) - I_zz->contents * q->contents *
    r->contents;
  sigma_6 = (((((((((((mc_a_tmp + hc_a_tmp * (sigma_6 * sigma_6) * n_a_tmp_tmp)
                      - ic_a_tmp * (sigma_7 * sigma_7) * o_a_tmp_tmp) + jc_a_tmp
                     * (ib_a_tmp_tmp * ib_a_tmp_tmp) * p_a_tmp_tmp) - kc_a_tmp *
                    (s_a * s_a) * q_a_tmp_tmp) + dc_a_tmp * (t_a * t_a) *
                   l_1->contents * j_a_tmp * ab_a_tmp_tmp) - ec_a_tmp * (u_a *
    u_a) * l_1->contents * m_a_tmp * bb_a_tmp_tmp) - fc_a_tmp * (v_a * v_a) *
                 l_2->contents * p_a_tmp * cb_a_tmp_tmp) + gc_a_tmp * (w_a * w_a)
                * l_2->contents * s_a_tmp * db_a_tmp_tmp) + dc_a_tmp * (x_a *
    x_a) * l_z->contents * j_a_tmp * r_a_tmp_tmp) + ec_a_tmp * (y_a * y_a) *
              l_z->contents * m_a_tmp * s_a_tmp_tmp) + fc_a_tmp * (ab_a * ab_a) *
             l_z->contents * p_a_tmp * t_a_tmp_tmp) + gc_a_tmp * (bb_a * bb_a) *
    l_z->contents * s_a_tmp * u_a_tmp_tmp;
  sigma_7 = I_xx->contents;
  u_a = a_tmp * a_tmp;
  m_a_tmp_tmp = u_a * gamma_quadratic_du->contents;
  cost_tmp_tmp = mb_a_tmp * mb_a_tmp;
  l_a_tmp_tmp = cost_tmp_tmp * gamma_quadratic_du->contents;
  b_cost_tmp_tmp = nb_a_tmp * nb_a_tmp;
  ib_a_tmp_tmp = b_cost_tmp_tmp * gamma_quadratic_du->contents;
  s_a = qb_a_tmp * qb_a_tmp;
  t_a = b_a_tmp * b_a_tmp;
  cost_tmp = f_a_tmp_tmp * f_a_tmp_tmp;
  b_cost_tmp = g_a_tmp_tmp * g_a_tmp_tmp;
  c_cost_tmp = cb_a_tmp * cb_a_tmp;
  d_cost_tmp = ob_a_tmp * ob_a_tmp;
  *cost = ((((((((((((((((m_a_tmp_tmp * (a * a) + m_a_tmp_tmp * (b_a * b_a)) +
    m_a_tmp_tmp * (c_a * c_a)) + m_a_tmp_tmp * (d_a * d_a)) + t_a *
                       gamma_quadratic_dv->contents * (e_a * e_a)) + l_a_tmp_tmp
                      * (f_a * f_a)) + l_a_tmp_tmp * (g_a * g_a)) + l_a_tmp_tmp *
                    (h_a * h_a)) + l_a_tmp_tmp * (i_a * i_a)) + ib_a_tmp_tmp *
                  (j_a * j_a)) + ib_a_tmp_tmp * (k_a * k_a)) + ib_a_tmp_tmp *
                (l_a * l_a)) + ib_a_tmp_tmp * (m_a * m_a)) + d_cost_tmp *
              gamma_quadratic_dv->contents * (n_a * n_a) / 10000.0) + s_a *
             gamma_quadratic_dv->contents * (o_a * o_a)) + c_cost_tmp *
            gamma_quadratic_dv->contents * (p_a * p_a) / (4.0 * (q_a * q_a))) +
           cost_tmp * gamma_quadratic_dv->contents * (r_a * r_a) / (x * x)) +
    b_cost_tmp * gamma_quadratic_dv->contents * (sigma_6 * sigma_6) / (sigma_7 *
    sigma_7);

  /*          sigma_1 = 2*I_zz*p*r - 2*I_xx*p*r - 2*I_yy*dv_global(5) + 2*K_p_T*u_in(1)^2*gain_motor^2*l_z*sin(u_in(5)*gain_el) + 2*K_p_T*u_in(2)^2*gain_motor^2*l_z*sin(u_in(6)*gain_el) + 2*K_p_T*u_in(3)^2*gain_motor^2*l_z*sin(u_in(7)*gain_el) + 2*K_p_T*u_in(4)^2*gain_motor^2*l_z*sin(u_in(8)*gain_el) - 2*K_p_M*u_in(1)^2*gain_motor^2*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az) + 2*K_p_M*u_in(2)^2*gain_motor^2*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az) - 2*K_p_M*u_in(3)^2*gain_motor^2*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az) + 2*K_p_M*u_in(4)^2*gain_motor^2*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az) + Cm_zero*S*V^2*rho*wing_chord + 2*K_p_T*u_in(1)^2*gain_motor^2*l_4*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az) + 2*K_p_T*u_in(2)^2*gain_motor^2*l_4*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az) - 2*K_p_T*u_in(3)^2*gain_motor^2*l_3*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az) - 2*K_p_T*u_in(4)^2*gain_motor^2*l_3*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az) + Cm_alpha*S*Theta*V^2*rho*wing_chord - Cm_alpha*S*V^2*flight_path_angle*rho*wing_chord; */
  /*            */
  /*          sigma_2 = I_zz*dv_global(6) - I_xx*p*q + I_yy*p*q + K_p_T*u_in(1)^2*gain_motor^2*l_1*sin(u_in(5)*gain_el) - K_p_T*u_in(2)^2*gain_motor^2*l_1*sin(u_in(6)*gain_el) - K_p_T*u_in(3)^2*gain_motor^2*l_2*sin(u_in(7)*gain_el) + K_p_T*u_in(4)^2*gain_motor^2*l_2*sin(u_in(8)*gain_el) - K_p_M*u_in(1)^2*gain_motor^2*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az) + K_p_M*u_in(2)^2*gain_motor^2*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az) - K_p_M*u_in(3)^2*gain_motor^2*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az) + K_p_M*u_in(4)^2*gain_motor^2*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az) - K_p_T*u_in(1)^2*gain_motor^2*l_4*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az) - K_p_T*u_in(2)^2*gain_motor^2*l_4*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az) + K_p_T*u_in(3)^2*gain_motor^2*l_3*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az) + K_p_T*u_in(4)^2*gain_motor^2*l_3*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az); */
  /*            */
  /*          sigma_3 = I_yy*q*r - I_xx*dv_global(4) - I_zz*q*r + K_p_M*u_in(1)^2*gain_motor^2*sin(u_in(5)*gain_el) - K_p_M*u_in(2)^2*gain_motor^2*sin(u_in(6)*gain_el) + K_p_M*u_in(3)^2*gain_motor^2*sin(u_in(7)*gain_el) - K_p_M*u_in(4)^2*gain_motor^2*sin(u_in(8)*gain_el) + K_p_T*u_in(1)^2*gain_motor^2*l_1*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az) - K_p_T*u_in(2)^2*gain_motor^2*l_1*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az) - K_p_T*u_in(3)^2*gain_motor^2*l_2*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az) + K_p_T*u_in(4)^2*gain_motor^2*l_2*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az) + K_p_T*u_in(1)^2*gain_motor^2*l_z*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az) + K_p_T*u_in(2)^2*gain_motor^2*l_z*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az) + K_p_T*u_in(3)^2*gain_motor^2*l_z*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az) + K_p_T*u_in(4)^2*gain_motor^2*l_z*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az); */
  /*            */
  /*          sigma_4 = (100*(K_p_T*gain_motor^2*(cos(Psi)*sin(Phi) - cos(Phi)*sin(Psi)*sin(Theta))*(sin(u_in(5)*gain_el)*u_in(1)^2 + sin(u_in(6)*gain_el)*u_in(2)^2 + sin(u_in(7)*gain_el)*u_in(3)^2 + sin(u_in(8)*gain_el)*u_in(4)^2) + K_p_T*gain_motor^2*cos(Theta)*sin(Phi)*(cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*u_in(4)^2) - K_p_T*gain_motor^2*cos(Phi)*cos(Theta)*(cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)*u_in(4)^2)))/m - 100*dv_global(3) + 981; */
  /*            */
  /*          sigma_5 = dv_global(2) - (K_p_T*gain_motor^2*(cos(Psi)*sin(Phi) - cos(Phi)*sin(Psi)*sin(Theta))*(cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)*u_in(4)^2) + K_p_T*gain_motor^2*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta))*(cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*u_in(4)^2) - K_p_T*gain_motor^2*cos(Theta)*sin(Psi)*(sin(u_in(5)*gain_el)*u_in(1)^2 + sin(u_in(6)*gain_el)*u_in(2)^2 + sin(u_in(7)*gain_el)*u_in(3)^2 + sin(u_in(8)*gain_el)*u_in(4)^2))/m; */
  /*            */
  /*          sigma_6 = dv_global(1) + (K_p_T*gain_motor^2*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta))*(cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)*u_in(4)^2) + K_p_T*gain_motor^2*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta))*(cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*u_in(4)^2) + K_p_T*gain_motor^2*cos(Psi)*cos(Theta)*(sin(u_in(5)*gain_el)*u_in(1)^2 + sin(u_in(6)*gain_el)*u_in(2)^2 + sin(u_in(7)*gain_el)*u_in(3)^2 + sin(u_in(8)*gain_el)*u_in(4)^2))/m; */
  /*            */
  /*          sigma_7 = gain_motor^2; */
  /*            */
  /*          sigma_8 = cos(Psi)*sin(Phi) - cos(Phi)*sin(Psi)*sin(Theta); */
  /*            */
  /*          computed_gradient = zeros(12,1);  */
  /*   */
  /*          computed_gradient(1) =                                                                           (2*W_dv_1^2*sigma_6*(2*K_p_T*u_in(1)*sigma_7*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta)) + 2*K_p_T*u_in(1)*sigma_7*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta)) + 2*K_p_T*u_in(1)*sigma_7*sin(u_in(5)*gain_el)*cos(Psi)*cos(Theta)))/m - (2*W_dv_2^2*sigma_5*(2*K_p_T*u_in(1)*sigma_7*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta)) - 2*K_p_T*u_in(1)*sigma_7*sin(u_in(5)*gain_el)*cos(Theta)*sin(Psi) + 2*K_p_T*u_in(1)*sigma_7*sigma_8*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)))/m - (2*W_act_motor^2*gamma_quadratic*(desired_motor_value - u_in(1)*gain_motor))/gain_motor - (4*u_in(1)*W_dv_6^2*sigma_2*sigma_7*(K_p_M*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az) - K_p_T*l_1*sin(u_in(5)*gain_el) + K_p_T*l_4*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)))/I_zz^2 + (4*u_in(1)*W_dv_4^2*sigma_3*sigma_7*(K_p_M*sin(u_in(5)*gain_el) + K_p_T*l_1*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az) + K_p_T*l_z*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)))/I_xx^2 + (2*u_in(1)*W_dv_5^2*sigma_1*sigma_7*(K_p_T*l_z*sin(u_in(5)*gain_el) - K_p_M*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az) + K_p_T*l_4*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)))/I_yy^2 + (K_p_T*u_in(1)*W_dv_3^2*sigma_4*sigma_7*(sin(u_in(5)*gain_el)*cos(Psi)*sin(Phi) + cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*cos(Theta)*sin(Phi) - sin(u_in(5)*gain_el)*cos(Phi)*sin(Psi)*sin(Theta) - cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)*cos(Phi)*cos(Theta)))/(25*m); */
  /*          computed_gradient(2) =                                                                           (2*W_dv_1^2*sigma_6*(2*K_p_T*u_in(2)*sigma_7*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta)) + 2*K_p_T*u_in(2)*sigma_7*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta)) + 2*K_p_T*u_in(2)*sigma_7*sin(u_in(6)*gain_el)*cos(Psi)*cos(Theta)))/m - (2*W_dv_2^2*sigma_5*(2*K_p_T*u_in(2)*sigma_7*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta)) - 2*K_p_T*u_in(2)*sigma_7*sin(u_in(6)*gain_el)*cos(Theta)*sin(Psi) + 2*K_p_T*u_in(2)*sigma_7*sigma_8*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)))/m - (2*W_act_motor^2*gamma_quadratic*(desired_motor_value - u_in(2)*gain_motor))/gain_motor - (4*u_in(2)*W_dv_6^2*sigma_2*sigma_7*(K_p_T*l_1*sin(u_in(6)*gain_el) - K_p_M*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az) + K_p_T*l_4*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)))/I_zz^2 - (4*u_in(2)*W_dv_4^2*sigma_3*sigma_7*(K_p_M*sin(u_in(6)*gain_el) + K_p_T*l_1*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az) - K_p_T*l_z*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)))/I_xx^2 + (2*u_in(2)*W_dv_5^2*sigma_1*sigma_7*(K_p_T*l_z*sin(u_in(6)*gain_el) + K_p_M*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az) + K_p_T*l_4*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)))/I_yy^2 + (K_p_T*u_in(2)*W_dv_3^2*sigma_4*sigma_7*(sin(u_in(6)*gain_el)*cos(Psi)*sin(Phi) + cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*cos(Theta)*sin(Phi) - sin(u_in(6)*gain_el)*cos(Phi)*sin(Psi)*sin(Theta) - cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)*cos(Phi)*cos(Theta)))/(25*m); */
  /*          computed_gradient(3) =                                                                           (2*W_dv_1^2*sigma_6*(2*K_p_T*u_in(3)*sigma_7*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta)) + 2*K_p_T*u_in(3)*sigma_7*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta)) + 2*K_p_T*u_in(3)*sigma_7*sin(u_in(7)*gain_el)*cos(Psi)*cos(Theta)))/m - (2*W_dv_2^2*sigma_5*(2*K_p_T*u_in(3)*sigma_7*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta)) - 2*K_p_T*u_in(3)*sigma_7*sin(u_in(7)*gain_el)*cos(Theta)*sin(Psi) + 2*K_p_T*u_in(3)*sigma_7*sigma_8*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)))/m - (2*W_act_motor^2*gamma_quadratic*(desired_motor_value - u_in(3)*gain_motor))/gain_motor - (4*u_in(3)*W_dv_6^2*sigma_2*sigma_7*(K_p_T*l_2*sin(u_in(7)*gain_el) + K_p_M*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az) - K_p_T*l_3*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)))/I_zz^2 + (4*u_in(3)*W_dv_4^2*sigma_3*sigma_7*(K_p_M*sin(u_in(7)*gain_el) - K_p_T*l_2*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az) + K_p_T*l_z*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)))/I_xx^2 - (2*u_in(3)*W_dv_5^2*sigma_1*sigma_7*(K_p_M*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az) - K_p_T*l_z*sin(u_in(7)*gain_el) + K_p_T*l_3*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)))/I_yy^2 + (K_p_T*u_in(3)*W_dv_3^2*sigma_4*sigma_7*(sin(u_in(7)*gain_el)*cos(Psi)*sin(Phi) + cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*cos(Theta)*sin(Phi) - sin(u_in(7)*gain_el)*cos(Phi)*sin(Psi)*sin(Theta) - cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)*cos(Phi)*cos(Theta)))/(25*m); */
  /*          computed_gradient(4) =                                                                           (2*W_dv_1^2*sigma_6*(2*K_p_T*u_in(4)*sigma_7*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta)) + 2*K_p_T*u_in(4)*sigma_7*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta)) + 2*K_p_T*u_in(4)*sigma_7*sin(u_in(8)*gain_el)*cos(Psi)*cos(Theta)))/m - (2*W_dv_2^2*sigma_5*(2*K_p_T*u_in(4)*sigma_7*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta)) - 2*K_p_T*u_in(4)*sigma_7*sin(u_in(8)*gain_el)*cos(Theta)*sin(Psi) + 2*K_p_T*u_in(4)*sigma_7*sigma_8*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)))/m - (2*W_act_motor^2*gamma_quadratic*(desired_motor_value - u_in(4)*gain_motor))/gain_motor + (4*u_in(4)*W_dv_6^2*sigma_2*sigma_7*(K_p_T*l_2*sin(u_in(8)*gain_el) + K_p_M*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az) + K_p_T*l_3*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)))/I_zz^2 + (4*u_in(4)*W_dv_4^2*sigma_3*sigma_7*(K_p_T*l_2*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az) - K_p_M*sin(u_in(8)*gain_el) + K_p_T*l_z*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)))/I_xx^2 + (2*u_in(4)*W_dv_5^2*sigma_1*sigma_7*(K_p_T*l_z*sin(u_in(8)*gain_el) + K_p_M*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az) - K_p_T*l_3*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)))/I_yy^2 + (K_p_T*u_in(4)*W_dv_3^2*sigma_4*sigma_7*(sin(u_in(8)*gain_el)*cos(Psi)*sin(Phi) + cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*cos(Theta)*sin(Phi) - sin(u_in(8)*gain_el)*cos(Phi)*sin(Psi)*sin(Theta) - cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)*cos(Phi)*cos(Theta)))/(25*m); */
  /*          computed_gradient(5) =(2*W_dv_2^2*sigma_5*(K_p_T*u_in(1)^2*gain_el*sigma_7*cos(u_in(5)*gain_el)*cos(Theta)*sin(Psi) + K_p_T*u_in(1)^2*gain_el*sigma_7*sin(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta)) + K_p_T*u_in(1)^2*gain_el*sigma_7*sigma_8*cos(u_in(9)*gain_az)*sin(u_in(5)*gain_el)))/m - (2*W_act_tilt_el^2*gamma_quadratic*(desired_el_value - u_in(5)*gain_el))/gain_el - (2*W_dv_1^2*sigma_6*(K_p_T*u_in(1)^2*gain_el*sigma_7*cos(u_in(9)*gain_az)*sin(u_in(5)*gain_el)*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta)) - K_p_T*u_in(1)^2*gain_el*sigma_7*cos(u_in(5)*gain_el)*cos(Psi)*cos(Theta) + K_p_T*u_in(1)^2*gain_el*sigma_7*sin(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta))))/m + (2*u_in(1)^2*W_dv_6^2*gain_el*sigma_2*sigma_7*(K_p_T*l_1*cos(u_in(5)*gain_el) + K_p_M*cos(u_in(9)*gain_az)*sin(u_in(5)*gain_el) + K_p_T*l_4*sin(u_in(5)*gain_el)*sin(u_in(9)*gain_az)))/I_zz^2 - (2*u_in(1)^2*W_dv_4^2*gain_el*sigma_3*sigma_7*(K_p_T*l_1*cos(u_in(9)*gain_az)*sin(u_in(5)*gain_el) - K_p_M*cos(u_in(5)*gain_el) + K_p_T*l_z*sin(u_in(5)*gain_el)*sin(u_in(9)*gain_az)))/I_xx^2 + (u_in(1)^2*W_dv_5^2*gain_el*sigma_1*sigma_7*(K_p_T*l_z*cos(u_in(5)*gain_el) + K_p_M*sin(u_in(5)*gain_el)*sin(u_in(9)*gain_az) - K_p_T*l_4*cos(u_in(9)*gain_az)*sin(u_in(5)*gain_el)))/I_yy^2 + (K_p_T*u_in(1)^2*W_dv_3^2*gain_el*sigma_4*sigma_7*(cos(u_in(5)*gain_el)*cos(Psi)*sin(Phi) - sin(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*cos(Theta)*sin(Phi) - cos(u_in(5)*gain_el)*cos(Phi)*sin(Psi)*sin(Theta) + cos(u_in(9)*gain_az)*sin(u_in(5)*gain_el)*cos(Phi)*cos(Theta)))/(50*m); */
  /*          computed_gradient(6) =(2*W_dv_2^2*sigma_5*(K_p_T*u_in(2)^2*gain_el*sigma_7*cos(u_in(6)*gain_el)*cos(Theta)*sin(Psi) + K_p_T*u_in(2)^2*gain_el*sigma_7*sin(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta)) + K_p_T*u_in(2)^2*gain_el*sigma_7*sigma_8*cos(u_in(10)*gain_az)*sin(u_in(6)*gain_el)))/m - (2*W_act_tilt_el^2*gamma_quadratic*(desired_el_value - u_in(6)*gain_el))/gain_el - (2*W_dv_1^2*sigma_6*(K_p_T*u_in(2)^2*gain_el*sigma_7*cos(u_in(10)*gain_az)*sin(u_in(6)*gain_el)*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta)) - K_p_T*u_in(2)^2*gain_el*sigma_7*cos(u_in(6)*gain_el)*cos(Psi)*cos(Theta) + K_p_T*u_in(2)^2*gain_el*sigma_7*sin(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta))))/m - (2*u_in(2)^2*W_dv_6^2*gain_el*sigma_2*sigma_7*(K_p_T*l_1*cos(u_in(6)*gain_el) + K_p_M*cos(u_in(10)*gain_az)*sin(u_in(6)*gain_el) - K_p_T*l_4*sin(u_in(6)*gain_el)*sin(u_in(10)*gain_az)))/I_zz^2 - (2*u_in(2)^2*W_dv_4^2*gain_el*sigma_3*sigma_7*(K_p_M*cos(u_in(6)*gain_el) - K_p_T*l_1*cos(u_in(10)*gain_az)*sin(u_in(6)*gain_el) + K_p_T*l_z*sin(u_in(6)*gain_el)*sin(u_in(10)*gain_az)))/I_xx^2 - (u_in(2)^2*W_dv_5^2*gain_el*sigma_1*sigma_7*(K_p_M*sin(u_in(6)*gain_el)*sin(u_in(10)*gain_az) - K_p_T*l_z*cos(u_in(6)*gain_el) + K_p_T*l_4*cos(u_in(10)*gain_az)*sin(u_in(6)*gain_el)))/I_yy^2 + (K_p_T*u_in(2)^2*W_dv_3^2*gain_el*sigma_4*sigma_7*(cos(u_in(6)*gain_el)*cos(Psi)*sin(Phi) - sin(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*cos(Theta)*sin(Phi) - cos(u_in(6)*gain_el)*cos(Phi)*sin(Psi)*sin(Theta) + cos(u_in(10)*gain_az)*sin(u_in(6)*gain_el)*cos(Phi)*cos(Theta)))/(50*m); */
  /*          computed_gradient(7) =(2*W_dv_2^2*sigma_5*(K_p_T*u_in(3)^2*gain_el*sigma_7*cos(u_in(7)*gain_el)*cos(Theta)*sin(Psi) + K_p_T*u_in(3)^2*gain_el*sigma_7*sin(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta)) + K_p_T*u_in(3)^2*gain_el*sigma_7*sigma_8*cos(u_in(11)*gain_az)*sin(u_in(7)*gain_el)))/m - (2*W_act_tilt_el^2*gamma_quadratic*(desired_el_value - u_in(7)*gain_el))/gain_el - (2*W_dv_1^2*sigma_6*(K_p_T*u_in(3)^2*gain_el*sigma_7*cos(u_in(11)*gain_az)*sin(u_in(7)*gain_el)*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta)) - K_p_T*u_in(3)^2*gain_el*sigma_7*cos(u_in(7)*gain_el)*cos(Psi)*cos(Theta) + K_p_T*u_in(3)^2*gain_el*sigma_7*sin(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta))))/m - (2*u_in(3)^2*W_dv_6^2*gain_el*sigma_2*sigma_7*(K_p_T*l_2*cos(u_in(7)*gain_el) - K_p_M*cos(u_in(11)*gain_az)*sin(u_in(7)*gain_el) + K_p_T*l_3*sin(u_in(7)*gain_el)*sin(u_in(11)*gain_az)))/I_zz^2 + (2*u_in(3)^2*W_dv_4^2*gain_el*sigma_3*sigma_7*(K_p_M*cos(u_in(7)*gain_el) + K_p_T*l_2*cos(u_in(11)*gain_az)*sin(u_in(7)*gain_el) - K_p_T*l_z*sin(u_in(7)*gain_el)*sin(u_in(11)*gain_az)))/I_xx^2 + (u_in(3)^2*W_dv_5^2*gain_el*sigma_1*sigma_7*(K_p_T*l_z*cos(u_in(7)*gain_el) + K_p_M*sin(u_in(7)*gain_el)*sin(u_in(11)*gain_az) + K_p_T*l_3*cos(u_in(11)*gain_az)*sin(u_in(7)*gain_el)))/I_yy^2 + (K_p_T*u_in(3)^2*W_dv_3^2*gain_el*sigma_4*sigma_7*(cos(u_in(7)*gain_el)*cos(Psi)*sin(Phi) - sin(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*cos(Theta)*sin(Phi) - cos(u_in(7)*gain_el)*cos(Phi)*sin(Psi)*sin(Theta) + cos(u_in(11)*gain_az)*sin(u_in(7)*gain_el)*cos(Phi)*cos(Theta)))/(50*m); */
  /*          computed_gradient(8) =(2*W_dv_2^2*sigma_5*(K_p_T*u_in(4)^2*gain_el*sigma_7*cos(u_in(8)*gain_el)*cos(Theta)*sin(Psi) + K_p_T*u_in(4)^2*gain_el*sigma_7*sin(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta)) + K_p_T*u_in(4)^2*gain_el*sigma_7*sigma_8*cos(u_in(12)*gain_az)*sin(u_in(8)*gain_el)))/m - (2*W_act_tilt_el^2*gamma_quadratic*(desired_el_value - u_in(8)*gain_el))/gain_el - (2*W_dv_1^2*sigma_6*(K_p_T*u_in(4)^2*gain_el*sigma_7*cos(u_in(12)*gain_az)*sin(u_in(8)*gain_el)*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta)) - K_p_T*u_in(4)^2*gain_el*sigma_7*cos(u_in(8)*gain_el)*cos(Psi)*cos(Theta) + K_p_T*u_in(4)^2*gain_el*sigma_7*sin(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta))))/m - (2*u_in(4)^2*W_dv_6^2*gain_el*sigma_2*sigma_7*(K_p_M*cos(u_in(12)*gain_az)*sin(u_in(8)*gain_el) - K_p_T*l_2*cos(u_in(8)*gain_el) + K_p_T*l_3*sin(u_in(8)*gain_el)*sin(u_in(12)*gain_az)))/I_zz^2 - (2*u_in(4)^2*W_dv_4^2*gain_el*sigma_3*sigma_7*(K_p_M*cos(u_in(8)*gain_el) + K_p_T*l_2*cos(u_in(12)*gain_az)*sin(u_in(8)*gain_el) + K_p_T*l_z*sin(u_in(8)*gain_el)*sin(u_in(12)*gain_az)))/I_xx^2 + (u_in(4)^2*W_dv_5^2*gain_el*sigma_1*sigma_7*(K_p_T*l_z*cos(u_in(8)*gain_el) - K_p_M*sin(u_in(8)*gain_el)*sin(u_in(12)*gain_az) + K_p_T*l_3*cos(u_in(12)*gain_az)*sin(u_in(8)*gain_el)))/I_yy^2 + (K_p_T*u_in(4)^2*W_dv_3^2*gain_el*sigma_4*sigma_7*(cos(u_in(8)*gain_el)*cos(Psi)*sin(Phi) - sin(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*cos(Theta)*sin(Phi) - cos(u_in(8)*gain_el)*cos(Phi)*sin(Psi)*sin(Theta) + cos(u_in(12)*gain_az)*sin(u_in(8)*gain_el)*cos(Phi)*cos(Theta)))/(50*m); */
  /*          computed_gradient(9) =                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              (2*W_dv_1^2*sigma_6*(K_p_T*u_in(1)^2*gain_az*sigma_7*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta)) - K_p_T*u_in(1)^2*gain_az*sigma_7*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta))))/m - (2*W_act_tilt_az^2*gamma_quadratic*(desired_az_value - u_in(9)*gain_az))/gain_az - (2*W_dv_2^2*sigma_5*(K_p_T*u_in(1)^2*gain_az*sigma_7*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta)) - K_p_T*u_in(1)^2*gain_az*sigma_7*sigma_8*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)))/m - (u_in(1)^2*W_dv_5^2*gain_az*sigma_1*sigma_7*cos(u_in(5)*gain_el)*(K_p_M*cos(u_in(9)*gain_az) + K_p_T*l_4*sin(u_in(9)*gain_az)))/I_yy^2 + (2*u_in(1)^2*W_dv_6^2*gain_az*sigma_2*sigma_7*cos(u_in(5)*gain_el)*(K_p_M*sin(u_in(9)*gain_az) - K_p_T*l_4*cos(u_in(9)*gain_az)))/I_zz^2 + (2*K_p_T*u_in(1)^2*W_dv_4^2*gain_az*sigma_3*sigma_7*cos(u_in(5)*gain_el)*(l_z*cos(u_in(9)*gain_az) - l_1*sin(u_in(9)*gain_az)))/I_xx^2 + (K_p_T*u_in(1)^2*W_dv_3^2*gain_az*sigma_4*sigma_7*sin(Phi + u_in(9)*gain_az)*cos(u_in(5)*gain_el)*cos(Theta))/(50*m); */
  /*          computed_gradient(10) =                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              (2*W_dv_1^2*sigma_6*(K_p_T*u_in(2)^2*gain_az*sigma_7*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta)) - K_p_T*u_in(2)^2*gain_az*sigma_7*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta))))/m - (2*W_act_tilt_az^2*gamma_quadratic*(desired_az_value - u_in(10)*gain_az))/gain_az - (2*W_dv_2^2*sigma_5*(K_p_T*u_in(2)^2*gain_az*sigma_7*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta)) - K_p_T*u_in(2)^2*gain_az*sigma_7*sigma_8*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)))/m + (u_in(2)^2*W_dv_5^2*gain_az*sigma_1*sigma_7*cos(u_in(6)*gain_el)*(K_p_M*cos(u_in(10)*gain_az) - K_p_T*l_4*sin(u_in(10)*gain_az)))/I_yy^2 - (2*u_in(2)^2*W_dv_6^2*gain_az*sigma_2*sigma_7*cos(u_in(6)*gain_el)*(K_p_M*sin(u_in(10)*gain_az) + K_p_T*l_4*cos(u_in(10)*gain_az)))/I_zz^2 + (2*K_p_T*u_in(2)^2*W_dv_4^2*gain_az*sigma_3*sigma_7*cos(u_in(6)*gain_el)*(l_z*cos(u_in(10)*gain_az) + l_1*sin(u_in(10)*gain_az)))/I_xx^2 + (K_p_T*u_in(2)^2*W_dv_3^2*gain_az*sigma_4*sigma_7*sin(Phi + u_in(10)*gain_az)*cos(u_in(6)*gain_el)*cos(Theta))/(50*m); */
  /*          computed_gradient(11) =                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              (2*W_dv_1^2*sigma_6*(K_p_T*u_in(3)^2*gain_az*sigma_7*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta)) - K_p_T*u_in(3)^2*gain_az*sigma_7*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta))))/m - (2*W_act_tilt_az^2*gamma_quadratic*(desired_az_value - u_in(11)*gain_az))/gain_az - (2*W_dv_2^2*sigma_5*(K_p_T*u_in(3)^2*gain_az*sigma_7*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta)) - K_p_T*u_in(3)^2*gain_az*sigma_7*sigma_8*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)))/m - (u_in(3)^2*W_dv_5^2*gain_az*sigma_1*sigma_7*cos(u_in(7)*gain_el)*(K_p_M*cos(u_in(11)*gain_az) - K_p_T*l_3*sin(u_in(11)*gain_az)))/I_yy^2 + (2*u_in(3)^2*W_dv_6^2*gain_az*sigma_2*sigma_7*cos(u_in(7)*gain_el)*(K_p_M*sin(u_in(11)*gain_az) + K_p_T*l_3*cos(u_in(11)*gain_az)))/I_zz^2 + (2*K_p_T*u_in(3)^2*W_dv_4^2*gain_az*sigma_3*sigma_7*cos(u_in(7)*gain_el)*(l_z*cos(u_in(11)*gain_az) + l_2*sin(u_in(11)*gain_az)))/I_xx^2 + (K_p_T*u_in(3)^2*W_dv_3^2*gain_az*sigma_4*sigma_7*sin(Phi + u_in(11)*gain_az)*cos(u_in(7)*gain_el)*cos(Theta))/(50*m); */
  /*          computed_gradient(12) =                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              (2*W_dv_1^2*sigma_6*(K_p_T*u_in(4)^2*gain_az*sigma_7*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta)) - K_p_T*u_in(4)^2*gain_az*sigma_7*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta))))/m - (2*W_act_tilt_az^2*gamma_quadratic*(desired_az_value - u_in(12)*gain_az))/gain_az - (2*W_dv_2^2*sigma_5*(K_p_T*u_in(4)^2*gain_az*sigma_7*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta)) - K_p_T*u_in(4)^2*gain_az*sigma_7*sigma_8*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)))/m + (u_in(4)^2*W_dv_5^2*gain_az*sigma_1*sigma_7*cos(u_in(8)*gain_el)*(K_p_M*cos(u_in(12)*gain_az) + K_p_T*l_3*sin(u_in(12)*gain_az)))/I_yy^2 - (2*u_in(4)^2*W_dv_6^2*gain_az*sigma_2*sigma_7*cos(u_in(8)*gain_el)*(K_p_M*sin(u_in(12)*gain_az) - K_p_T*l_3*cos(u_in(12)*gain_az)))/I_zz^2 + (2*K_p_T*u_in(4)^2*W_dv_4^2*gain_az*sigma_3*sigma_7*cos(u_in(8)*gain_el)*(l_z*cos(u_in(12)*gain_az) - l_2*sin(u_in(12)*gain_az)))/I_xx^2 + (K_p_T*u_in(4)^2*W_dv_3^2*gain_az*sigma_4*sigma_7*sin(Phi + u_in(12)*gain_az)*cos(u_in(8)*gain_el)*cos(Theta))/(50*m); */
  /*          %No aerodynamic on Forces */
  /*          sigma_1 = 2*I_zz*p*r - 2*I_xx*p*r - 2*I_yy*dv_global(5) + 2*K_p_T*u_in(1)^2*gain_motor^2*l_z*sin(u_in(5)*gain_el) + 2*K_p_T*u_in(2)^2*gain_motor^2*l_z*sin(u_in(6)*gain_el) + 2*K_p_T*u_in(3)^2*gain_motor^2*l_z*sin(u_in(7)*gain_el) + 2*K_p_T*u_in(4)^2*gain_motor^2*l_z*sin(u_in(8)*gain_el) - 2*K_p_M*u_in(1)^2*gain_motor^2*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az) + 2*K_p_M*u_in(2)^2*gain_motor^2*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az) - 2*K_p_M*u_in(3)^2*gain_motor^2*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az) + 2*K_p_M*u_in(4)^2*gain_motor^2*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az) + Cm_zero*S*V^2*rho*wing_chord + 2*K_p_T*u_in(1)^2*gain_motor^2*l_4*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az) + 2*K_p_T*u_in(2)^2*gain_motor^2*l_4*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az) - 2*K_p_T*u_in(3)^2*gain_motor^2*l_3*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az) - 2*K_p_T*u_in(4)^2*gain_motor^2*l_3*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az) + Cm_alpha*S*Theta*V^2*rho*wing_chord - Cm_alpha*S*V^2*flight_path_angle*rho*wing_chord;  */
  /*           */
  /*          sigma_2 = I_zz*dv_global(6) - I_xx*p*q + I_yy*p*q + K_p_T*u_in(1)^2*gain_motor^2*l_1*sin(u_in(5)*gain_el) - K_p_T*u_in(2)^2*gain_motor^2*l_1*sin(u_in(6)*gain_el) - K_p_T*u_in(3)^2*gain_motor^2*l_2*sin(u_in(7)*gain_el) + K_p_T*u_in(4)^2*gain_motor^2*l_2*sin(u_in(8)*gain_el) - K_p_M*u_in(1)^2*gain_motor^2*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az) + K_p_M*u_in(2)^2*gain_motor^2*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az) - K_p_M*u_in(3)^2*gain_motor^2*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az) + K_p_M*u_in(4)^2*gain_motor^2*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az) - K_p_T*u_in(1)^2*gain_motor^2*l_4*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az) - K_p_T*u_in(2)^2*gain_motor^2*l_4*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az) + K_p_T*u_in(3)^2*gain_motor^2*l_3*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az) + K_p_T*u_in(4)^2*gain_motor^2*l_3*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az); */
  /*            */
  /*          sigma_3 = I_yy*q*r - I_xx*dv_global(4) - I_zz*q*r + K_p_M*u_in(1)^2*gain_motor^2*sin(u_in(5)*gain_el) - K_p_M*u_in(2)^2*gain_motor^2*sin(u_in(6)*gain_el) + K_p_M*u_in(3)^2*gain_motor^2*sin(u_in(7)*gain_el) - K_p_M*u_in(4)^2*gain_motor^2*sin(u_in(8)*gain_el) + K_p_T*u_in(1)^2*gain_motor^2*l_1*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az) - K_p_T*u_in(2)^2*gain_motor^2*l_1*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az) - K_p_T*u_in(3)^2*gain_motor^2*l_2*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az) + K_p_T*u_in(4)^2*gain_motor^2*l_2*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az) + K_p_T*u_in(1)^2*gain_motor^2*l_z*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az) + K_p_T*u_in(2)^2*gain_motor^2*l_z*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az) + K_p_T*u_in(3)^2*gain_motor^2*l_z*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az) + K_p_T*u_in(4)^2*gain_motor^2*l_z*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az); */
  /*            */
  /*          sigma_4 = (100*(K_p_T*gain_motor^2*(cos(Psi)*sin(Phi) - cos(Phi)*sin(Psi)*sin(Theta))*(sin(u_in(5)*gain_el)*u_in(1)^2 + sin(u_in(6)*gain_el)*u_in(2)^2 + sin(u_in(7)*gain_el)*u_in(3)^2 + sin(u_in(8)*gain_el)*u_in(4)^2) + K_p_T*gain_motor^2*cos(Theta)*sin(Phi)*(cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*u_in(4)^2) - K_p_T*gain_motor^2*cos(Phi)*cos(Theta)*(cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)*u_in(4)^2)))/m - 100*dv_global(3) + 981; */
  /*            */
  /*          sigma_5 = dv_global(2) - (K_p_T*gain_motor^2*(cos(Psi)*sin(Phi) - cos(Phi)*sin(Psi)*sin(Theta))*(cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)*u_in(4)^2) + K_p_T*gain_motor^2*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta))*(cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*u_in(4)^2) - K_p_T*gain_motor^2*cos(Theta)*sin(Psi)*(sin(u_in(5)*gain_el)*u_in(1)^2 + sin(u_in(6)*gain_el)*u_in(2)^2 + sin(u_in(7)*gain_el)*u_in(3)^2 + sin(u_in(8)*gain_el)*u_in(4)^2))/m; */
  /*            */
  /*          sigma_6 = dv_global(1) + (K_p_T*gain_motor^2*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta))*(cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)*u_in(4)^2) + K_p_T*gain_motor^2*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta))*(cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*u_in(4)^2) + K_p_T*gain_motor^2*cos(Psi)*cos(Theta)*(sin(u_in(5)*gain_el)*u_in(1)^2 + sin(u_in(6)*gain_el)*u_in(2)^2 + sin(u_in(7)*gain_el)*u_in(3)^2 + sin(u_in(8)*gain_el)*u_in(4)^2))/m; */
  /*            */
  /*          sigma_7 = gain_motor^2; */
  /*            */
  /*          sigma_8 = cos(Psi)*sin(Phi) - cos(Phi)*sin(Psi)*sin(Theta); */
  /*            */
  /*          computed_gradient = zeros(12,1);  */
  /*          computed_gradient(1) =                                                                           (2*W_dv_1^2*gamma_quadratic_dv*sigma_6*(2*K_p_T*u_in(1)*sigma_7*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta)) + 2*K_p_T*u_in(1)*sigma_7*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta)) + 2*K_p_T*u_in(1)*sigma_7*sin(u_in(5)*gain_el)*cos(Psi)*cos(Theta)))/m - (2*W_dv_2^2*gamma_quadratic_dv*sigma_5*(2*K_p_T*u_in(1)*sigma_7*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta)) - 2*K_p_T*u_in(1)*sigma_7*sin(u_in(5)*gain_el)*cos(Theta)*sin(Psi) + 2*K_p_T*u_in(1)*sigma_7*sigma_8*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)))/m - (2*W_act_motor^2*gamma_quadratic_du*(desired_motor_value - u_in(1)*gain_motor))/gain_motor - (4*u_in(1)*W_dv_6^2*gamma_quadratic_dv*sigma_2*sigma_7*(K_p_M*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az) - K_p_T*l_1*sin(u_in(5)*gain_el) + K_p_T*l_4*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)))/I_zz^2 + (4*u_in(1)*W_dv_4^2*gamma_quadratic_dv*sigma_3*sigma_7*(K_p_M*sin(u_in(5)*gain_el) + K_p_T*l_1*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az) + K_p_T*l_z*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)))/I_xx^2 + (2*u_in(1)*W_dv_5^2*gamma_quadratic_dv*sigma_1*sigma_7*(K_p_T*l_z*sin(u_in(5)*gain_el) - K_p_M*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az) + K_p_T*l_4*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)))/I_yy^2 + (K_p_T*u_in(1)*W_dv_3^2*gamma_quadratic_dv*sigma_4*sigma_7*(sin(u_in(5)*gain_el)*cos(Psi)*sin(Phi) + cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*cos(Theta)*sin(Phi) - sin(u_in(5)*gain_el)*cos(Phi)*sin(Psi)*sin(Theta) - cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)*cos(Phi)*cos(Theta)))/(25*m); */
  /*          computed_gradient(2) =                                                                           (2*W_dv_1^2*gamma_quadratic_dv*sigma_6*(2*K_p_T*u_in(2)*sigma_7*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta)) + 2*K_p_T*u_in(2)*sigma_7*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta)) + 2*K_p_T*u_in(2)*sigma_7*sin(u_in(6)*gain_el)*cos(Psi)*cos(Theta)))/m - (2*W_dv_2^2*gamma_quadratic_dv*sigma_5*(2*K_p_T*u_in(2)*sigma_7*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta)) - 2*K_p_T*u_in(2)*sigma_7*sin(u_in(6)*gain_el)*cos(Theta)*sin(Psi) + 2*K_p_T*u_in(2)*sigma_7*sigma_8*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)))/m - (2*W_act_motor^2*gamma_quadratic_du*(desired_motor_value - u_in(2)*gain_motor))/gain_motor - (4*u_in(2)*W_dv_6^2*gamma_quadratic_dv*sigma_2*sigma_7*(K_p_T*l_1*sin(u_in(6)*gain_el) - K_p_M*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az) + K_p_T*l_4*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)))/I_zz^2 - (4*u_in(2)*W_dv_4^2*gamma_quadratic_dv*sigma_3*sigma_7*(K_p_M*sin(u_in(6)*gain_el) + K_p_T*l_1*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az) - K_p_T*l_z*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)))/I_xx^2 + (2*u_in(2)*W_dv_5^2*gamma_quadratic_dv*sigma_1*sigma_7*(K_p_T*l_z*sin(u_in(6)*gain_el) + K_p_M*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az) + K_p_T*l_4*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)))/I_yy^2 + (K_p_T*u_in(2)*W_dv_3^2*gamma_quadratic_dv*sigma_4*sigma_7*(sin(u_in(6)*gain_el)*cos(Psi)*sin(Phi) + cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*cos(Theta)*sin(Phi) - sin(u_in(6)*gain_el)*cos(Phi)*sin(Psi)*sin(Theta) - cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)*cos(Phi)*cos(Theta)))/(25*m); */
  /*          computed_gradient(3) =                                                                           (2*W_dv_1^2*gamma_quadratic_dv*sigma_6*(2*K_p_T*u_in(3)*sigma_7*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta)) + 2*K_p_T*u_in(3)*sigma_7*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta)) + 2*K_p_T*u_in(3)*sigma_7*sin(u_in(7)*gain_el)*cos(Psi)*cos(Theta)))/m - (2*W_dv_2^2*gamma_quadratic_dv*sigma_5*(2*K_p_T*u_in(3)*sigma_7*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta)) - 2*K_p_T*u_in(3)*sigma_7*sin(u_in(7)*gain_el)*cos(Theta)*sin(Psi) + 2*K_p_T*u_in(3)*sigma_7*sigma_8*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)))/m - (2*W_act_motor^2*gamma_quadratic_du*(desired_motor_value - u_in(3)*gain_motor))/gain_motor - (4*u_in(3)*W_dv_6^2*gamma_quadratic_dv*sigma_2*sigma_7*(K_p_T*l_2*sin(u_in(7)*gain_el) + K_p_M*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az) - K_p_T*l_3*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)))/I_zz^2 + (4*u_in(3)*W_dv_4^2*gamma_quadratic_dv*sigma_3*sigma_7*(K_p_M*sin(u_in(7)*gain_el) - K_p_T*l_2*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az) + K_p_T*l_z*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)))/I_xx^2 - (2*u_in(3)*W_dv_5^2*gamma_quadratic_dv*sigma_1*sigma_7*(K_p_M*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az) - K_p_T*l_z*sin(u_in(7)*gain_el) + K_p_T*l_3*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)))/I_yy^2 + (K_p_T*u_in(3)*W_dv_3^2*gamma_quadratic_dv*sigma_4*sigma_7*(sin(u_in(7)*gain_el)*cos(Psi)*sin(Phi) + cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*cos(Theta)*sin(Phi) - sin(u_in(7)*gain_el)*cos(Phi)*sin(Psi)*sin(Theta) - cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)*cos(Phi)*cos(Theta)))/(25*m); */
  /*          computed_gradient(4) =                                                                           (2*W_dv_1^2*gamma_quadratic_dv*sigma_6*(2*K_p_T*u_in(4)*sigma_7*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta)) + 2*K_p_T*u_in(4)*sigma_7*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta)) + 2*K_p_T*u_in(4)*sigma_7*sin(u_in(8)*gain_el)*cos(Psi)*cos(Theta)))/m - (2*W_dv_2^2*gamma_quadratic_dv*sigma_5*(2*K_p_T*u_in(4)*sigma_7*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta)) - 2*K_p_T*u_in(4)*sigma_7*sin(u_in(8)*gain_el)*cos(Theta)*sin(Psi) + 2*K_p_T*u_in(4)*sigma_7*sigma_8*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)))/m - (2*W_act_motor^2*gamma_quadratic_du*(desired_motor_value - u_in(4)*gain_motor))/gain_motor + (4*u_in(4)*W_dv_6^2*gamma_quadratic_dv*sigma_2*sigma_7*(K_p_T*l_2*sin(u_in(8)*gain_el) + K_p_M*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az) + K_p_T*l_3*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)))/I_zz^2 + (4*u_in(4)*W_dv_4^2*gamma_quadratic_dv*sigma_3*sigma_7*(K_p_T*l_2*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az) - K_p_M*sin(u_in(8)*gain_el) + K_p_T*l_z*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)))/I_xx^2 + (2*u_in(4)*W_dv_5^2*gamma_quadratic_dv*sigma_1*sigma_7*(K_p_T*l_z*sin(u_in(8)*gain_el) + K_p_M*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az) - K_p_T*l_3*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)))/I_yy^2 + (K_p_T*u_in(4)*W_dv_3^2*gamma_quadratic_dv*sigma_4*sigma_7*(sin(u_in(8)*gain_el)*cos(Psi)*sin(Phi) + cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*cos(Theta)*sin(Phi) - sin(u_in(8)*gain_el)*cos(Phi)*sin(Psi)*sin(Theta) - cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)*cos(Phi)*cos(Theta)))/(25*m); */
  /*          computed_gradient(5) =(2*W_dv_2^2*gamma_quadratic_dv*sigma_5*(K_p_T*u_in(1)^2*gain_el*sigma_7*cos(u_in(5)*gain_el)*cos(Theta)*sin(Psi) + K_p_T*u_in(1)^2*gain_el*sigma_7*sin(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta)) + K_p_T*u_in(1)^2*gain_el*sigma_7*sigma_8*cos(u_in(9)*gain_az)*sin(u_in(5)*gain_el)))/m - (2*W_act_tilt_el^2*gamma_quadratic_du*(desired_el_value - u_in(5)*gain_el))/gain_el - (2*W_dv_1^2*gamma_quadratic_dv*sigma_6*(K_p_T*u_in(1)^2*gain_el*sigma_7*cos(u_in(9)*gain_az)*sin(u_in(5)*gain_el)*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta)) - K_p_T*u_in(1)^2*gain_el*sigma_7*cos(u_in(5)*gain_el)*cos(Psi)*cos(Theta) + K_p_T*u_in(1)^2*gain_el*sigma_7*sin(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta))))/m + (2*u_in(1)^2*W_dv_6^2*gain_el*gamma_quadratic_dv*sigma_2*sigma_7*(K_p_T*l_1*cos(u_in(5)*gain_el) + K_p_M*cos(u_in(9)*gain_az)*sin(u_in(5)*gain_el) + K_p_T*l_4*sin(u_in(5)*gain_el)*sin(u_in(9)*gain_az)))/I_zz^2 - (2*u_in(1)^2*W_dv_4^2*gain_el*gamma_quadratic_dv*sigma_3*sigma_7*(K_p_T*l_1*cos(u_in(9)*gain_az)*sin(u_in(5)*gain_el) - K_p_M*cos(u_in(5)*gain_el) + K_p_T*l_z*sin(u_in(5)*gain_el)*sin(u_in(9)*gain_az)))/I_xx^2 + (u_in(1)^2*W_dv_5^2*gain_el*gamma_quadratic_dv*sigma_1*sigma_7*(K_p_T*l_z*cos(u_in(5)*gain_el) + K_p_M*sin(u_in(5)*gain_el)*sin(u_in(9)*gain_az) - K_p_T*l_4*cos(u_in(9)*gain_az)*sin(u_in(5)*gain_el)))/I_yy^2 + (K_p_T*u_in(1)^2*W_dv_3^2*gain_el*gamma_quadratic_dv*sigma_4*sigma_7*(cos(u_in(5)*gain_el)*cos(Psi)*sin(Phi) - sin(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*cos(Theta)*sin(Phi) - cos(u_in(5)*gain_el)*cos(Phi)*sin(Psi)*sin(Theta) + cos(u_in(9)*gain_az)*sin(u_in(5)*gain_el)*cos(Phi)*cos(Theta)))/(50*m); */
  /*          computed_gradient(6) =(2*W_dv_2^2*gamma_quadratic_dv*sigma_5*(K_p_T*u_in(2)^2*gain_el*sigma_7*cos(u_in(6)*gain_el)*cos(Theta)*sin(Psi) + K_p_T*u_in(2)^2*gain_el*sigma_7*sin(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta)) + K_p_T*u_in(2)^2*gain_el*sigma_7*sigma_8*cos(u_in(10)*gain_az)*sin(u_in(6)*gain_el)))/m - (2*W_act_tilt_el^2*gamma_quadratic_du*(desired_el_value - u_in(6)*gain_el))/gain_el - (2*W_dv_1^2*gamma_quadratic_dv*sigma_6*(K_p_T*u_in(2)^2*gain_el*sigma_7*cos(u_in(10)*gain_az)*sin(u_in(6)*gain_el)*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta)) - K_p_T*u_in(2)^2*gain_el*sigma_7*cos(u_in(6)*gain_el)*cos(Psi)*cos(Theta) + K_p_T*u_in(2)^2*gain_el*sigma_7*sin(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta))))/m - (2*u_in(2)^2*W_dv_6^2*gain_el*gamma_quadratic_dv*sigma_2*sigma_7*(K_p_T*l_1*cos(u_in(6)*gain_el) + K_p_M*cos(u_in(10)*gain_az)*sin(u_in(6)*gain_el) - K_p_T*l_4*sin(u_in(6)*gain_el)*sin(u_in(10)*gain_az)))/I_zz^2 - (2*u_in(2)^2*W_dv_4^2*gain_el*gamma_quadratic_dv*sigma_3*sigma_7*(K_p_M*cos(u_in(6)*gain_el) - K_p_T*l_1*cos(u_in(10)*gain_az)*sin(u_in(6)*gain_el) + K_p_T*l_z*sin(u_in(6)*gain_el)*sin(u_in(10)*gain_az)))/I_xx^2 - (u_in(2)^2*W_dv_5^2*gain_el*gamma_quadratic_dv*sigma_1*sigma_7*(K_p_M*sin(u_in(6)*gain_el)*sin(u_in(10)*gain_az) - K_p_T*l_z*cos(u_in(6)*gain_el) + K_p_T*l_4*cos(u_in(10)*gain_az)*sin(u_in(6)*gain_el)))/I_yy^2 + (K_p_T*u_in(2)^2*W_dv_3^2*gain_el*gamma_quadratic_dv*sigma_4*sigma_7*(cos(u_in(6)*gain_el)*cos(Psi)*sin(Phi) - sin(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*cos(Theta)*sin(Phi) - cos(u_in(6)*gain_el)*cos(Phi)*sin(Psi)*sin(Theta) + cos(u_in(10)*gain_az)*sin(u_in(6)*gain_el)*cos(Phi)*cos(Theta)))/(50*m); */
  /*          computed_gradient(7) =(2*W_dv_2^2*gamma_quadratic_dv*sigma_5*(K_p_T*u_in(3)^2*gain_el*sigma_7*cos(u_in(7)*gain_el)*cos(Theta)*sin(Psi) + K_p_T*u_in(3)^2*gain_el*sigma_7*sin(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta)) + K_p_T*u_in(3)^2*gain_el*sigma_7*sigma_8*cos(u_in(11)*gain_az)*sin(u_in(7)*gain_el)))/m - (2*W_act_tilt_el^2*gamma_quadratic_du*(desired_el_value - u_in(7)*gain_el))/gain_el - (2*W_dv_1^2*gamma_quadratic_dv*sigma_6*(K_p_T*u_in(3)^2*gain_el*sigma_7*cos(u_in(11)*gain_az)*sin(u_in(7)*gain_el)*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta)) - K_p_T*u_in(3)^2*gain_el*sigma_7*cos(u_in(7)*gain_el)*cos(Psi)*cos(Theta) + K_p_T*u_in(3)^2*gain_el*sigma_7*sin(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta))))/m - (2*u_in(3)^2*W_dv_6^2*gain_el*gamma_quadratic_dv*sigma_2*sigma_7*(K_p_T*l_2*cos(u_in(7)*gain_el) - K_p_M*cos(u_in(11)*gain_az)*sin(u_in(7)*gain_el) + K_p_T*l_3*sin(u_in(7)*gain_el)*sin(u_in(11)*gain_az)))/I_zz^2 + (2*u_in(3)^2*W_dv_4^2*gain_el*gamma_quadratic_dv*sigma_3*sigma_7*(K_p_M*cos(u_in(7)*gain_el) + K_p_T*l_2*cos(u_in(11)*gain_az)*sin(u_in(7)*gain_el) - K_p_T*l_z*sin(u_in(7)*gain_el)*sin(u_in(11)*gain_az)))/I_xx^2 + (u_in(3)^2*W_dv_5^2*gain_el*gamma_quadratic_dv*sigma_1*sigma_7*(K_p_T*l_z*cos(u_in(7)*gain_el) + K_p_M*sin(u_in(7)*gain_el)*sin(u_in(11)*gain_az) + K_p_T*l_3*cos(u_in(11)*gain_az)*sin(u_in(7)*gain_el)))/I_yy^2 + (K_p_T*u_in(3)^2*W_dv_3^2*gain_el*gamma_quadratic_dv*sigma_4*sigma_7*(cos(u_in(7)*gain_el)*cos(Psi)*sin(Phi) - sin(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*cos(Theta)*sin(Phi) - cos(u_in(7)*gain_el)*cos(Phi)*sin(Psi)*sin(Theta) + cos(u_in(11)*gain_az)*sin(u_in(7)*gain_el)*cos(Phi)*cos(Theta)))/(50*m); */
  /*          computed_gradient(8) =(2*W_dv_2^2*gamma_quadratic_dv*sigma_5*(K_p_T*u_in(4)^2*gain_el*sigma_7*cos(u_in(8)*gain_el)*cos(Theta)*sin(Psi) + K_p_T*u_in(4)^2*gain_el*sigma_7*sin(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta)) + K_p_T*u_in(4)^2*gain_el*sigma_7*sigma_8*cos(u_in(12)*gain_az)*sin(u_in(8)*gain_el)))/m - (2*W_act_tilt_el^2*gamma_quadratic_du*(desired_el_value - u_in(8)*gain_el))/gain_el - (2*W_dv_1^2*gamma_quadratic_dv*sigma_6*(K_p_T*u_in(4)^2*gain_el*sigma_7*cos(u_in(12)*gain_az)*sin(u_in(8)*gain_el)*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta)) - K_p_T*u_in(4)^2*gain_el*sigma_7*cos(u_in(8)*gain_el)*cos(Psi)*cos(Theta) + K_p_T*u_in(4)^2*gain_el*sigma_7*sin(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta))))/m - (2*u_in(4)^2*W_dv_6^2*gain_el*gamma_quadratic_dv*sigma_2*sigma_7*(K_p_M*cos(u_in(12)*gain_az)*sin(u_in(8)*gain_el) - K_p_T*l_2*cos(u_in(8)*gain_el) + K_p_T*l_3*sin(u_in(8)*gain_el)*sin(u_in(12)*gain_az)))/I_zz^2 - (2*u_in(4)^2*W_dv_4^2*gain_el*gamma_quadratic_dv*sigma_3*sigma_7*(K_p_M*cos(u_in(8)*gain_el) + K_p_T*l_2*cos(u_in(12)*gain_az)*sin(u_in(8)*gain_el) + K_p_T*l_z*sin(u_in(8)*gain_el)*sin(u_in(12)*gain_az)))/I_xx^2 + (u_in(4)^2*W_dv_5^2*gain_el*gamma_quadratic_dv*sigma_1*sigma_7*(K_p_T*l_z*cos(u_in(8)*gain_el) - K_p_M*sin(u_in(8)*gain_el)*sin(u_in(12)*gain_az) + K_p_T*l_3*cos(u_in(12)*gain_az)*sin(u_in(8)*gain_el)))/I_yy^2 + (K_p_T*u_in(4)^2*W_dv_3^2*gain_el*gamma_quadratic_dv*sigma_4*sigma_7*(cos(u_in(8)*gain_el)*cos(Psi)*sin(Phi) - sin(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*cos(Theta)*sin(Phi) - cos(u_in(8)*gain_el)*cos(Phi)*sin(Psi)*sin(Theta) + cos(u_in(12)*gain_az)*sin(u_in(8)*gain_el)*cos(Phi)*cos(Theta)))/(50*m); */
  /*          computed_gradient(9) =                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              (2*W_dv_1^2*gamma_quadratic_dv*sigma_6*(K_p_T*u_in(1)^2*gain_az*sigma_7*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta)) - K_p_T*u_in(1)^2*gain_az*sigma_7*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta))))/m - (2*W_dv_2^2*gamma_quadratic_dv*sigma_5*(K_p_T*u_in(1)^2*gain_az*sigma_7*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta)) - K_p_T*u_in(1)^2*gain_az*sigma_7*sigma_8*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)))/m - (2*W_act_tilt_az^2*gamma_quadratic_du*(desired_az_value - u_in(9)*gain_az))/gain_az - (u_in(1)^2*W_dv_5^2*gain_az*gamma_quadratic_dv*sigma_1*sigma_7*cos(u_in(5)*gain_el)*(K_p_M*cos(u_in(9)*gain_az) + K_p_T*l_4*sin(u_in(9)*gain_az)))/I_yy^2 + (2*u_in(1)^2*W_dv_6^2*gain_az*gamma_quadratic_dv*sigma_2*sigma_7*cos(u_in(5)*gain_el)*(K_p_M*sin(u_in(9)*gain_az) - K_p_T*l_4*cos(u_in(9)*gain_az)))/I_zz^2 + (2*K_p_T*u_in(1)^2*W_dv_4^2*gain_az*gamma_quadratic_dv*sigma_3*sigma_7*cos(u_in(5)*gain_el)*(l_z*cos(u_in(9)*gain_az) - l_1*sin(u_in(9)*gain_az)))/I_xx^2 + (K_p_T*u_in(1)^2*W_dv_3^2*gain_az*gamma_quadratic_dv*sigma_4*sigma_7*sin(Phi + u_in(9)*gain_az)*cos(u_in(5)*gain_el)*cos(Theta))/(50*m); */
  /*          computed_gradient(10) =                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              (2*W_dv_1^2*gamma_quadratic_dv*sigma_6*(K_p_T*u_in(2)^2*gain_az*sigma_7*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta)) - K_p_T*u_in(2)^2*gain_az*sigma_7*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta))))/m - (2*W_dv_2^2*gamma_quadratic_dv*sigma_5*(K_p_T*u_in(2)^2*gain_az*sigma_7*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta)) - K_p_T*u_in(2)^2*gain_az*sigma_7*sigma_8*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)))/m - (2*W_act_tilt_az^2*gamma_quadratic_du*(desired_az_value - u_in(10)*gain_az))/gain_az + (u_in(2)^2*W_dv_5^2*gain_az*gamma_quadratic_dv*sigma_1*sigma_7*cos(u_in(6)*gain_el)*(K_p_M*cos(u_in(10)*gain_az) - K_p_T*l_4*sin(u_in(10)*gain_az)))/I_yy^2 - (2*u_in(2)^2*W_dv_6^2*gain_az*gamma_quadratic_dv*sigma_2*sigma_7*cos(u_in(6)*gain_el)*(K_p_M*sin(u_in(10)*gain_az) + K_p_T*l_4*cos(u_in(10)*gain_az)))/I_zz^2 + (2*K_p_T*u_in(2)^2*W_dv_4^2*gain_az*gamma_quadratic_dv*sigma_3*sigma_7*cos(u_in(6)*gain_el)*(l_z*cos(u_in(10)*gain_az) + l_1*sin(u_in(10)*gain_az)))/I_xx^2 + (K_p_T*u_in(2)^2*W_dv_3^2*gain_az*gamma_quadratic_dv*sigma_4*sigma_7*sin(Phi + u_in(10)*gain_az)*cos(u_in(6)*gain_el)*cos(Theta))/(50*m); */
  /*          computed_gradient(11) =                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              (2*W_dv_1^2*gamma_quadratic_dv*sigma_6*(K_p_T*u_in(3)^2*gain_az*sigma_7*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta)) - K_p_T*u_in(3)^2*gain_az*sigma_7*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta))))/m - (2*W_dv_2^2*gamma_quadratic_dv*sigma_5*(K_p_T*u_in(3)^2*gain_az*sigma_7*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta)) - K_p_T*u_in(3)^2*gain_az*sigma_7*sigma_8*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)))/m - (2*W_act_tilt_az^2*gamma_quadratic_du*(desired_az_value - u_in(11)*gain_az))/gain_az - (u_in(3)^2*W_dv_5^2*gain_az*gamma_quadratic_dv*sigma_1*sigma_7*cos(u_in(7)*gain_el)*(K_p_M*cos(u_in(11)*gain_az) - K_p_T*l_3*sin(u_in(11)*gain_az)))/I_yy^2 + (2*u_in(3)^2*W_dv_6^2*gain_az*gamma_quadratic_dv*sigma_2*sigma_7*cos(u_in(7)*gain_el)*(K_p_M*sin(u_in(11)*gain_az) + K_p_T*l_3*cos(u_in(11)*gain_az)))/I_zz^2 + (2*K_p_T*u_in(3)^2*W_dv_4^2*gain_az*gamma_quadratic_dv*sigma_3*sigma_7*cos(u_in(7)*gain_el)*(l_z*cos(u_in(11)*gain_az) + l_2*sin(u_in(11)*gain_az)))/I_xx^2 + (K_p_T*u_in(3)^2*W_dv_3^2*gain_az*gamma_quadratic_dv*sigma_4*sigma_7*sin(Phi + u_in(11)*gain_az)*cos(u_in(7)*gain_el)*cos(Theta))/(50*m); */
  /*          computed_gradient(12) =                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              (2*W_dv_1^2*gamma_quadratic_dv*sigma_6*(K_p_T*u_in(4)^2*gain_az*sigma_7*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta)) - K_p_T*u_in(4)^2*gain_az*sigma_7*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta))))/m - (2*W_dv_2^2*gamma_quadratic_dv*sigma_5*(K_p_T*u_in(4)^2*gain_az*sigma_7*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta)) - K_p_T*u_in(4)^2*gain_az*sigma_7*sigma_8*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)))/m - (2*W_act_tilt_az^2*gamma_quadratic_du*(desired_az_value - u_in(12)*gain_az))/gain_az + (u_in(4)^2*W_dv_5^2*gain_az*gamma_quadratic_dv*sigma_1*sigma_7*cos(u_in(8)*gain_el)*(K_p_M*cos(u_in(12)*gain_az) + K_p_T*l_3*sin(u_in(12)*gain_az)))/I_yy^2 - (2*u_in(4)^2*W_dv_6^2*gain_az*gamma_quadratic_dv*sigma_2*sigma_7*cos(u_in(8)*gain_el)*(K_p_M*sin(u_in(12)*gain_az) - K_p_T*l_3*cos(u_in(12)*gain_az)))/I_zz^2 + (2*K_p_T*u_in(4)^2*W_dv_4^2*gain_az*gamma_quadratic_dv*sigma_3*sigma_7*cos(u_in(8)*gain_el)*(l_z*cos(u_in(12)*gain_az) - l_2*sin(u_in(12)*gain_az)))/I_xx^2 + (K_p_T*u_in(4)^2*W_dv_3^2*gain_az*gamma_quadratic_dv*sigma_4*sigma_7*sin(Phi + u_in(12)*gain_az)*cos(u_in(8)*gain_el)*cos(Theta))/(50*m);      */
  ib_a_tmp_tmp = c_a_tmp;
  if (!b) {
    if (c_a_tmp < 0.0) {
      ib_a_tmp_tmp = -1.0;
    } else {
      ib_a_tmp_tmp = (c_a_tmp > 0.0);
    }
  }

  a = gain_motor->contents;
  b_a = gain_motor->contents;
  c_a = gain_motor->contents;
  l_a_tmp_tmp = ((n_a_tmp_tmp * (u_in[0] * u_in[0]) + o_a_tmp_tmp * (u_in[1] *
    u_in[1])) + p_a_tmp_tmp * (u_in[2] * u_in[2])) + q_a_tmp_tmp * (u_in[3] *
    u_in[3]);
  d_a = gain_motor->contents;
  e_a = gain_motor->contents;
  f_a = gain_motor->contents;
  g_a = gain_motor->contents;
  h_a = gain_motor->contents;
  i_a = gain_motor->contents;
  sigma_3 = (100.0 * dv_global->contents[2] + 100.0 * (((((sigma_3 -
    ib_a_tmp_tmp * gb_a_tmp * h_a_tmp) - K_p_T->contents * (g_a * g_a) * h_a_tmp
    * l_a_tmp_tmp) - K_p_T->contents * (h_a * h_a) * ab_a_tmp * c_a_tmp_tmp *
    db_a_tmp) + K_p_T->contents * (i_a * i_a) * a_tmp_tmp * ab_a_tmp * eb_a_tmp)
              + pb_a_tmp) / m->contents) - 981.0;
  g_a = gain_motor->contents;
  h_a = gain_motor->contents;
  i_a = gain_motor->contents;
  j_a = gain_motor->contents;
  k_a = gain_motor->contents;
  l_a = gain_motor->contents;
  m_a = gain_motor->contents;
  n_a = gain_motor->contents;
  o_a = gain_motor->contents;
  p_a = gain_motor->contents;
  q_a = gain_motor->contents;
  r_a = gain_motor->contents;
  sigma_4 = ((((((((((((((yb_a_tmp + ub_a_tmp * (g_a * g_a) * l_z->contents *
    n_a_tmp_tmp) + vb_a_tmp * (h_a * h_a) * l_z->contents * o_a_tmp_tmp) +
                        wb_a_tmp * (i_a * i_a) * l_z->contents * p_a_tmp_tmp) +
                       xb_a_tmp * (j_a * j_a) * l_z->contents * q_a_tmp_tmp) -
                      ac_a_tmp * (k_a * k_a) * j_a_tmp * r_a_tmp_tmp) + sigma_5 *
                     (l_a * l_a) * m_a_tmp * s_a_tmp_tmp) - sigma_4 * (m_a * m_a)
                    * p_a_tmp * t_a_tmp_tmp) + jb_a_tmp * (n_a * n_a) * s_a_tmp *
                   u_a_tmp_tmp) + bc_a_tmp) + ub_a_tmp * (o_a * o_a) *
                 l_4->contents * j_a_tmp * ab_a_tmp_tmp) + vb_a_tmp * (p_a * p_a)
                * l_4->contents * m_a_tmp * bb_a_tmp_tmp) - wb_a_tmp * (q_a *
    q_a) * l_3->contents * p_a_tmp * cb_a_tmp_tmp) - xb_a_tmp * (r_a * r_a) *
              l_3->contents * s_a_tmp * db_a_tmp_tmp) + cc_a_tmp) - hb_a_tmp;
  g_a = gain_motor->contents;
  h_a = gain_motor->contents;
  i_a = gain_motor->contents;
  j_a = gain_motor->contents;
  k_a = gain_motor->contents;
  l_a = gain_motor->contents;
  m_a = gain_motor->contents;
  n_a = gain_motor->contents;
  o_a = gain_motor->contents;
  p_a = gain_motor->contents;
  q_a = gain_motor->contents;
  r_a = gain_motor->contents;
  sigma_5 = (((((((((((lc_a_tmp + dc_a_tmp * (g_a * g_a) * l_1->contents *
                       n_a_tmp_tmp) - ec_a_tmp * (h_a * h_a) * l_1->contents *
                      o_a_tmp_tmp) - fc_a_tmp * (i_a * i_a) * l_2->contents *
                     p_a_tmp_tmp) + gc_a_tmp * (j_a * j_a) * l_2->contents *
                    q_a_tmp_tmp) - hc_a_tmp * (k_a * k_a) * j_a_tmp *
                   ab_a_tmp_tmp) + ic_a_tmp * (l_a * l_a) * m_a_tmp *
                  bb_a_tmp_tmp) - jc_a_tmp * (m_a * m_a) * p_a_tmp *
                 cb_a_tmp_tmp) + kc_a_tmp * (n_a * n_a) * s_a_tmp * db_a_tmp_tmp)
               - dc_a_tmp * (o_a * o_a) * l_4->contents * j_a_tmp * r_a_tmp_tmp)
              - ec_a_tmp * (p_a * p_a) * l_4->contents * m_a_tmp * s_a_tmp_tmp)
             + fc_a_tmp * (q_a * q_a) * l_3->contents * p_a_tmp * t_a_tmp_tmp) +
    gc_a_tmp * (r_a * r_a) * l_3->contents * s_a_tmp * u_a_tmp_tmp;
  g_a = gain_motor->contents;
  h_a = gain_motor->contents;
  i_a = gain_motor->contents;
  j_a = gain_motor->contents;
  k_a = gain_motor->contents;
  l_a = gain_motor->contents;
  m_a = gain_motor->contents;
  n_a = gain_motor->contents;
  o_a = gain_motor->contents;
  p_a = gain_motor->contents;
  q_a = gain_motor->contents;
  r_a = gain_motor->contents;
  sigma_6 = (((((((((((mc_a_tmp + hc_a_tmp * (g_a * g_a) * n_a_tmp_tmp) -
                      ic_a_tmp * (h_a * h_a) * o_a_tmp_tmp) + jc_a_tmp * (i_a *
    i_a) * p_a_tmp_tmp) - kc_a_tmp * (j_a * j_a) * q_a_tmp_tmp) + dc_a_tmp *
                   (k_a * k_a) * l_1->contents * j_a_tmp * ab_a_tmp_tmp) -
                  ec_a_tmp * (l_a * l_a) * l_1->contents * m_a_tmp *
                  bb_a_tmp_tmp) - fc_a_tmp * (m_a * m_a) * l_2->contents *
                 p_a_tmp * cb_a_tmp_tmp) + gc_a_tmp * (n_a * n_a) *
                l_2->contents * s_a_tmp * db_a_tmp_tmp) + dc_a_tmp * (o_a * o_a)
               * l_z->contents * j_a_tmp * r_a_tmp_tmp) + ec_a_tmp * (p_a * p_a)
              * l_z->contents * m_a_tmp * s_a_tmp_tmp) + fc_a_tmp * (q_a * q_a) *
             l_z->contents * p_a_tmp * t_a_tmp_tmp) + gc_a_tmp * (r_a * r_a) *
    l_z->contents * s_a_tmp * u_a_tmp_tmp;
  g_a = gain_motor->contents;
  sigma_7 = g_a * g_a;
  g_a = I_zz->contents;
  h_a = I_xx->contents;
  i_a = I_yy->contents;
  b_a_tmp = ib_a_tmp * u_in[0] * sigma_7;
  cb_a_tmp = K_p_T->contents * l_1->contents;
  ob_a_tmp = K_p_T->contents * l_z->contents;
  yb_a_tmp = K_p_T->contents * l_4->contents;
  qb_a_tmp = 25.0 * m->contents;
  ac_a_tmp = K_p_M->contents * cos(u_in[4] * gain_el->contents);
  nb_a_tmp = 2.0 * s_a * gamma_quadratic_dv->contents * (dv_global->contents[0]
    + ((((((S->contents * (c_a_tmp * c_a_tmp) * rho->contents * u_a_tmp *
            y_a_tmp * (((K_Cd->contents * (d_a_tmp * d_a_tmp) * (e_a_tmp *
    e_a_tmp) - w_a_tmp * (d_a_tmp * d_a_tmp) * Theta->contents *
    flight_path_angle->contents) + K_Cd->contents * (d_a_tmp * d_a_tmp) *
                        (f_a_tmp * f_a_tmp)) + Cd_zero->contents) / 2.0 +
            x_a_tmp * (c_a_tmp * c_a_tmp) * rho->contents * v_a_tmp * g_a_tmp /
            2.0) * (sin(Phi->contents) * sin(Psi->contents) + cos(Phi->contents)
                    * cos(Psi->contents) * e_a_tmp_tmp) + K_p_T->contents * (d_a
    * d_a) * rb_a_tmp * eb_a_tmp) + tb_a_tmp * ib_a_tmp_tmp * gb_a_tmp) +
         K_p_T->contents * (e_a * e_a) * (cos(Phi->contents) * sin(Psi->contents)
    - cos(Psi->contents) * sin(Phi->contents) * e_a_tmp_tmp) * db_a_tmp) +
        K_p_T->contents * (f_a * f_a) * b_a_tmp_tmp * ab_a_tmp * l_a_tmp_tmp) -
       fb_a_tmp) / m->contents);
  mb_a_tmp = 2.0 * t_a * gamma_quadratic_dv->contents * (dv_global->contents[1]
    - ((((((S->contents * (c_a_tmp * c_a_tmp) * rho->contents * u_a_tmp *
            y_a_tmp * (((K_Cd->contents * (d_a_tmp * d_a_tmp) * (e_a_tmp *
    e_a_tmp) - w_a_tmp * (d_a_tmp * d_a_tmp) * Theta->contents *
    flight_path_angle->contents) + K_Cd->contents * (d_a_tmp * d_a_tmp) *
                        (f_a_tmp * f_a_tmp)) + Cd_zero->contents) / 2.0 +
            x_a_tmp * (c_a_tmp * c_a_tmp) * rho->contents * v_a_tmp * g_a_tmp /
            2.0) * (b_a_tmp_tmp * c_a_tmp_tmp - a_tmp_tmp * d_a_tmp_tmp *
                    e_a_tmp_tmp) + K_p_T->contents * (a * a) * h_a_tmp *
           (((j_a_tmp * ab_a_tmp_tmp * (u_in[0] * u_in[0]) + m_a_tmp *
              bb_a_tmp_tmp * (u_in[1] * u_in[1])) + p_a_tmp * cb_a_tmp_tmp *
             (u_in[2] * u_in[2])) + s_a_tmp * db_a_tmp_tmp * (u_in[3] * u_in[3])))
          + K_p_T->contents * (b_a * b_a) * (a_tmp_tmp * b_a_tmp_tmp +
    c_a_tmp_tmp * d_a_tmp_tmp * e_a_tmp_tmp) * (((j_a_tmp * r_a_tmp_tmp * (u_in
    [0] * u_in[0]) + m_a_tmp * s_a_tmp_tmp * (u_in[1] * u_in[1])) + p_a_tmp *
    t_a_tmp_tmp * (u_in[2] * u_in[2])) + s_a_tmp * u_a_tmp_tmp * (u_in[3] *
    u_in[3]))) - kb_a_tmp * ib_a_tmp_tmp * gb_a_tmp) - K_p_T->contents * (c_a *
    c_a) * ab_a_tmp * d_a_tmp_tmp * l_a_tmp_tmp) - lb_a_tmp) / m->contents);
  a_tmp = 2.0 * u_a * gamma_quadratic_du->contents;
  computed_gradient[0] = (((((nb_a_tmp * ((b_a_tmp * j_a_tmp * ab_a_tmp_tmp *
    rb_a_tmp + 2.0 * K_p_T->contents * u_in[0] * sigma_7 * cos(u_in[4] *
    gain_el->contents) * r_a_tmp_tmp * sb_a_tmp) + b_a_tmp * n_a_tmp_tmp *
    b_a_tmp_tmp * ab_a_tmp) / m->contents - mb_a_tmp * ((2.0 * K_p_T->contents *
    u_in[0] * sigma_7 * cos(u_in[4] * gain_el->contents) * sin(u_in[8] *
    gain_az->contents) * bb_a_tmp - 2.0 * K_p_T->contents * u_in[0] * sigma_7 *
    sin(u_in[4] * gain_el->contents) * ab_a_tmp * d_a_tmp_tmp) + b_a_tmp *
    h_a_tmp * j_a_tmp * ab_a_tmp_tmp) / m->contents) - a_tmp *
    (desired_motor_value->contents - u_in[0] * gain_motor->contents) /
    gain_motor->contents) - 4.0 * u_in[0] * cost_tmp *
    gamma_quadratic_dv->contents * sigma_5 * sigma_7 * ((K_p_M->contents *
    j_a_tmp * ab_a_tmp_tmp - cb_a_tmp * n_a_tmp_tmp) + yb_a_tmp * j_a_tmp *
    r_a_tmp_tmp) / (g_a * g_a)) + 4.0 * u_in[0] * b_cost_tmp *
    gamma_quadratic_dv->contents * sigma_6 * sigma_7 * ((K_p_M->contents *
    n_a_tmp_tmp + cb_a_tmp * j_a_tmp * ab_a_tmp_tmp) + ob_a_tmp * j_a_tmp *
    r_a_tmp_tmp) / (h_a * h_a)) + 2.0 * u_in[0] * c_cost_tmp *
    gamma_quadratic_dv->contents * sigma_4 * sigma_7 * ((ob_a_tmp * n_a_tmp_tmp
    - ac_a_tmp * r_a_tmp_tmp) + K_p_T->contents * l_4->contents * cos(u_in[4] *
    gain_el->contents) * ab_a_tmp_tmp) / (i_a * i_a)) - K_p_T->contents * u_in[0]
    * d_cost_tmp * gamma_quadratic_dv->contents * sigma_3 * sigma_7 *
    (((n_a_tmp_tmp * b_a_tmp_tmp * c_a_tmp_tmp + v_a_tmp_tmp * ab_a_tmp *
       c_a_tmp_tmp) - n_a_tmp_tmp * a_tmp_tmp * d_a_tmp_tmp * e_a_tmp_tmp) -
     eb_a_tmp_tmp * a_tmp_tmp * ab_a_tmp) / qb_a_tmp;
  a = I_zz->contents;
  b_a = I_xx->contents;
  c_a = I_yy->contents;
  b_a_tmp = ib_a_tmp * u_in[1] * sigma_7;
  bb_a = K_p_M->contents * cos(u_in[5] * gain_el->contents);
  computed_gradient[1] = (((((nb_a_tmp * ((b_a_tmp * m_a_tmp * bb_a_tmp_tmp *
    rb_a_tmp + 2.0 * K_p_T->contents * u_in[1] * sigma_7 * cos(u_in[5] *
    gain_el->contents) * s_a_tmp_tmp * sb_a_tmp) + b_a_tmp * o_a_tmp_tmp *
    b_a_tmp_tmp * ab_a_tmp) / m->contents - mb_a_tmp * ((2.0 * K_p_T->contents *
    u_in[1] * sigma_7 * cos(u_in[5] * gain_el->contents) * sin(u_in[9] *
    gain_az->contents) * bb_a_tmp - 2.0 * K_p_T->contents * u_in[1] * sigma_7 *
    sin(u_in[5] * gain_el->contents) * ab_a_tmp * d_a_tmp_tmp) + b_a_tmp *
    h_a_tmp * m_a_tmp * bb_a_tmp_tmp) / m->contents) - a_tmp *
    (desired_motor_value->contents - u_in[1] * gain_motor->contents) /
    gain_motor->contents) - 4.0 * u_in[1] * cost_tmp *
    gamma_quadratic_dv->contents * sigma_5 * sigma_7 * ((cb_a_tmp * o_a_tmp_tmp
    - K_p_M->contents * m_a_tmp * bb_a_tmp_tmp) + yb_a_tmp * m_a_tmp *
    s_a_tmp_tmp) / (a * a)) - 4.0 * u_in[1] * b_cost_tmp *
    gamma_quadratic_dv->contents * sigma_6 * sigma_7 * ((K_p_M->contents *
    o_a_tmp_tmp + cb_a_tmp * m_a_tmp * bb_a_tmp_tmp) - ob_a_tmp * m_a_tmp *
    s_a_tmp_tmp) / (b_a * b_a)) + 2.0 * u_in[1] * c_cost_tmp *
    gamma_quadratic_dv->contents * sigma_4 * sigma_7 * ((ob_a_tmp * o_a_tmp_tmp
    + bb_a * s_a_tmp_tmp) + K_p_T->contents * l_4->contents * cos(u_in[5] *
    gain_el->contents) * bb_a_tmp_tmp) / (c_a * c_a)) - K_p_T->contents * u_in[1]
    * d_cost_tmp * gamma_quadratic_dv->contents * sigma_3 * sigma_7 *
    (((o_a_tmp_tmp * b_a_tmp_tmp * c_a_tmp_tmp + w_a_tmp_tmp * ab_a_tmp *
       c_a_tmp_tmp) - o_a_tmp_tmp * a_tmp_tmp * d_a_tmp_tmp * e_a_tmp_tmp) -
     fb_a_tmp_tmp * a_tmp_tmp * ab_a_tmp) / qb_a_tmp;
  a = I_zz->contents;
  b_a = I_xx->contents;
  c_a = I_yy->contents;
  b_a_tmp = ib_a_tmp * u_in[2] * sigma_7;
  x_a = K_p_T->contents * l_2->contents;
  y_a = K_p_T->contents * l_3->contents;
  ab_a = K_p_M->contents * cos(u_in[6] * gain_el->contents);
  computed_gradient[2] = (((((nb_a_tmp * ((b_a_tmp * p_a_tmp * cb_a_tmp_tmp *
    rb_a_tmp + 2.0 * K_p_T->contents * u_in[2] * sigma_7 * cos(u_in[6] *
    gain_el->contents) * t_a_tmp_tmp * sb_a_tmp) + b_a_tmp * p_a_tmp_tmp *
    b_a_tmp_tmp * ab_a_tmp) / m->contents - mb_a_tmp * ((2.0 * K_p_T->contents *
    u_in[2] * sigma_7 * cos(u_in[6] * gain_el->contents) * sin(u_in[10] *
    gain_az->contents) * bb_a_tmp - 2.0 * K_p_T->contents * u_in[2] * sigma_7 *
    sin(u_in[6] * gain_el->contents) * ab_a_tmp * d_a_tmp_tmp) + b_a_tmp *
    h_a_tmp * p_a_tmp * cb_a_tmp_tmp) / m->contents) - a_tmp *
    (desired_motor_value->contents - u_in[2] * gain_motor->contents) /
    gain_motor->contents) - 4.0 * u_in[2] * cost_tmp *
    gamma_quadratic_dv->contents * sigma_5 * sigma_7 * ((x_a * p_a_tmp_tmp +
    K_p_M->contents * p_a_tmp * cb_a_tmp_tmp) - y_a * p_a_tmp * t_a_tmp_tmp) /
    (a * a)) + 4.0 * u_in[2] * b_cost_tmp * gamma_quadratic_dv->contents *
    sigma_6 * sigma_7 * ((K_p_M->contents * p_a_tmp_tmp - x_a * p_a_tmp *
    cb_a_tmp_tmp) + ob_a_tmp * p_a_tmp * t_a_tmp_tmp) / (b_a * b_a)) - 2.0 *
    u_in[2] * c_cost_tmp * gamma_quadratic_dv->contents * sigma_4 * sigma_7 *
    ((ab_a * t_a_tmp_tmp - ob_a_tmp * p_a_tmp_tmp) + K_p_T->contents *
     l_3->contents * cos(u_in[6] * gain_el->contents) * cb_a_tmp_tmp) / (c_a *
    c_a)) - K_p_T->contents * u_in[2] * d_cost_tmp *
    gamma_quadratic_dv->contents * sigma_3 * sigma_7 * (((p_a_tmp_tmp *
    b_a_tmp_tmp * c_a_tmp_tmp + x_a_tmp_tmp * ab_a_tmp * c_a_tmp_tmp) -
    p_a_tmp_tmp * a_tmp_tmp * d_a_tmp_tmp * e_a_tmp_tmp) - gb_a_tmp_tmp *
    a_tmp_tmp * ab_a_tmp) / qb_a_tmp;
  a = I_zz->contents;
  b_a = I_xx->contents;
  c_a = I_yy->contents;
  b_a_tmp = ib_a_tmp * u_in[3] * sigma_7;
  w_a = K_p_M->contents * cos(u_in[7] * gain_el->contents);
  computed_gradient[3] = (((((nb_a_tmp * ((b_a_tmp * s_a_tmp * db_a_tmp_tmp *
    rb_a_tmp + 2.0 * K_p_T->contents * u_in[3] * sigma_7 * cos(u_in[7] *
    gain_el->contents) * u_a_tmp_tmp * sb_a_tmp) + b_a_tmp * q_a_tmp_tmp *
    b_a_tmp_tmp * ab_a_tmp) / m->contents - mb_a_tmp * ((2.0 * K_p_T->contents *
    u_in[3] * sigma_7 * cos(u_in[7] * gain_el->contents) * sin(u_in[11] *
    gain_az->contents) * bb_a_tmp - 2.0 * K_p_T->contents * u_in[3] * sigma_7 *
    sin(u_in[7] * gain_el->contents) * ab_a_tmp * d_a_tmp_tmp) + b_a_tmp *
    h_a_tmp * s_a_tmp * db_a_tmp_tmp) / m->contents) - a_tmp *
    (desired_motor_value->contents - u_in[3] * gain_motor->contents) /
    gain_motor->contents) + 4.0 * u_in[3] * cost_tmp *
    gamma_quadratic_dv->contents * sigma_5 * sigma_7 * ((x_a * q_a_tmp_tmp +
    K_p_M->contents * s_a_tmp * db_a_tmp_tmp) + y_a * s_a_tmp * u_a_tmp_tmp) /
    (a * a)) + 4.0 * u_in[3] * b_cost_tmp * gamma_quadratic_dv->contents *
    sigma_6 * sigma_7 * ((x_a * s_a_tmp * db_a_tmp_tmp - K_p_M->contents *
    q_a_tmp_tmp) + ob_a_tmp * s_a_tmp * u_a_tmp_tmp) / (b_a * b_a)) + 2.0 *
    u_in[3] * c_cost_tmp * gamma_quadratic_dv->contents * sigma_4 * sigma_7 *
    ((ob_a_tmp * q_a_tmp_tmp + w_a * u_a_tmp_tmp) - K_p_T->contents *
     l_3->contents * cos(u_in[7] * gain_el->contents) * db_a_tmp_tmp) / (c_a *
    c_a)) - K_p_T->contents * u_in[3] * d_cost_tmp *
    gamma_quadratic_dv->contents * sigma_3 * sigma_7 * (((q_a_tmp_tmp *
    b_a_tmp_tmp * c_a_tmp_tmp + y_a_tmp_tmp * ab_a_tmp * c_a_tmp_tmp) -
    q_a_tmp_tmp * a_tmp_tmp * d_a_tmp_tmp * e_a_tmp_tmp) - hb_a_tmp_tmp *
    a_tmp_tmp * ab_a_tmp) / qb_a_tmp;
  a = I_zz->contents;
  b_a = I_xx->contents;
  c_a = I_yy->contents;
  b_a_tmp = dc_a_tmp * gain_el->contents * sigma_7;
  ob_a_tmp = b_a_tmp * j_a_tmp;
  qb_a_tmp = b_a_tmp * n_a_tmp_tmp * r_a_tmp_tmp;
  a_tmp = 2.0 * k_a_tmp;
  v_a = 50.0 * m->contents;
  ib_a_tmp_tmp = 2.0 * cost_tmp_tmp * gamma_quadratic_du->contents;
  s_a = k_a_tmp * c_cost_tmp;
  t_a = a_tmp * cost_tmp;
  u_a = dc_a_tmp * d_cost_tmp;
  computed_gradient[4] = (((((mb_a_tmp * ((ob_a_tmp * ab_a_tmp * d_a_tmp_tmp +
    qb_a_tmp * bb_a_tmp) + b_a_tmp * h_a_tmp * ab_a_tmp_tmp * n_a_tmp_tmp) /
    m->contents - ib_a_tmp_tmp * (desired_el_value->contents - h_a_tmp_tmp) /
    gain_el->contents) - nb_a_tmp * ((b_a_tmp * ab_a_tmp_tmp * n_a_tmp_tmp *
    rb_a_tmp - ob_a_tmp * b_a_tmp_tmp * ab_a_tmp) + qb_a_tmp * sb_a_tmp) /
    m->contents) + t_a * gain_el->contents * gamma_quadratic_dv->contents *
    sigma_5 * sigma_7 * ((K_p_T->contents * l_1->contents * cos(u_in[4] *
    gain_el->contents) + K_p_M->contents * ab_a_tmp_tmp * n_a_tmp_tmp) +
    yb_a_tmp * n_a_tmp_tmp * r_a_tmp_tmp) / (a * a)) - a_tmp * b_cost_tmp *
    gain_el->contents * gamma_quadratic_dv->contents * sigma_6 * sigma_7 *
    ((cb_a_tmp * ab_a_tmp_tmp * n_a_tmp_tmp - ac_a_tmp) + K_p_T->contents *
     l_z->contents * sin(u_in[4] * gain_el->contents) * r_a_tmp_tmp) / (b_a *
    b_a)) + s_a * gain_el->contents * gamma_quadratic_dv->contents * sigma_4 *
    sigma_7 * ((K_p_T->contents * l_z->contents * cos(u_in[4] *
    gain_el->contents) + K_p_M->contents * sin(u_in[4] * gain_el->contents) *
                r_a_tmp_tmp) - yb_a_tmp * ab_a_tmp_tmp * n_a_tmp_tmp) / (c_a *
    c_a)) - u_a * gain_el->contents * gamma_quadratic_dv->contents * sigma_3 *
    sigma_7 * (((j_a_tmp * b_a_tmp_tmp * c_a_tmp_tmp - n_a_tmp_tmp * r_a_tmp_tmp
                 * ab_a_tmp * c_a_tmp_tmp) - j_a_tmp * a_tmp_tmp * d_a_tmp_tmp *
                e_a_tmp_tmp) + ab_a_tmp_tmp * n_a_tmp_tmp * a_tmp_tmp * ab_a_tmp)
    / v_a;
  a = I_zz->contents;
  b_a = I_xx->contents;
  c_a = I_yy->contents;
  b_a_tmp = ec_a_tmp * gain_el->contents * sigma_7;
  ob_a_tmp = b_a_tmp * m_a_tmp;
  qb_a_tmp = b_a_tmp * o_a_tmp_tmp * s_a_tmp_tmp;
  ac_a_tmp = 2.0 * n_a_tmp;
  a_tmp = n_a_tmp * c_cost_tmp;
  g_a_tmp_tmp = ac_a_tmp * cost_tmp;
  x = ec_a_tmp * d_cost_tmp;
  computed_gradient[5] = (((((mb_a_tmp * ((ob_a_tmp * ab_a_tmp * d_a_tmp_tmp +
    qb_a_tmp * bb_a_tmp) + b_a_tmp * h_a_tmp * bb_a_tmp_tmp * o_a_tmp_tmp) /
    m->contents - ib_a_tmp_tmp * (desired_el_value->contents - i_a_tmp_tmp) /
    gain_el->contents) - nb_a_tmp * ((b_a_tmp * bb_a_tmp_tmp * o_a_tmp_tmp *
    rb_a_tmp - ob_a_tmp * b_a_tmp_tmp * ab_a_tmp) + qb_a_tmp * sb_a_tmp) /
    m->contents) - g_a_tmp_tmp * gain_el->contents *
    gamma_quadratic_dv->contents * sigma_5 * sigma_7 * ((K_p_T->contents *
    l_1->contents * cos(u_in[5] * gain_el->contents) + K_p_M->contents *
    bb_a_tmp_tmp * o_a_tmp_tmp) - yb_a_tmp * o_a_tmp_tmp * s_a_tmp_tmp) / (a * a))
    - ac_a_tmp * b_cost_tmp * gain_el->contents * gamma_quadratic_dv->contents *
    sigma_6 * sigma_7 * ((bb_a - cb_a_tmp * bb_a_tmp_tmp * o_a_tmp_tmp) +
    K_p_T->contents * l_z->contents * sin(u_in[5] * gain_el->contents) *
    s_a_tmp_tmp) / (b_a * b_a)) - a_tmp * gain_el->contents *
    gamma_quadratic_dv->contents * sigma_4 * sigma_7 * ((K_p_M->contents * sin
    (u_in[5] * gain_el->contents) * s_a_tmp_tmp - K_p_T->contents *
    l_z->contents * cos(u_in[5] * gain_el->contents)) + yb_a_tmp * bb_a_tmp_tmp *
    o_a_tmp_tmp) / (c_a * c_a)) - x * gain_el->contents *
    gamma_quadratic_dv->contents * sigma_3 * sigma_7 * (((m_a_tmp * b_a_tmp_tmp *
    c_a_tmp_tmp - o_a_tmp_tmp * s_a_tmp_tmp * ab_a_tmp * c_a_tmp_tmp) - m_a_tmp *
    a_tmp_tmp * d_a_tmp_tmp * e_a_tmp_tmp) + bb_a_tmp_tmp * o_a_tmp_tmp *
    a_tmp_tmp * ab_a_tmp) / v_a;
  a = I_zz->contents;
  b_a = I_xx->contents;
  c_a = I_yy->contents;
  b_a_tmp = fc_a_tmp * gain_el->contents * sigma_7;
  cb_a_tmp = b_a_tmp * p_a_tmp;
  ob_a_tmp = b_a_tmp * p_a_tmp_tmp * t_a_tmp_tmp;
  qb_a_tmp = 2.0 * q_a_tmp;
  ac_a_tmp = q_a_tmp * c_cost_tmp;
  bb_a = qb_a_tmp * cost_tmp;
  f_a_tmp_tmp = fc_a_tmp * d_cost_tmp;
  computed_gradient[6] = (((((mb_a_tmp * ((cb_a_tmp * ab_a_tmp * d_a_tmp_tmp +
    ob_a_tmp * bb_a_tmp) + b_a_tmp * h_a_tmp * cb_a_tmp_tmp * p_a_tmp_tmp) /
    m->contents - ib_a_tmp_tmp * (desired_el_value->contents - j_a_tmp_tmp) /
    gain_el->contents) - nb_a_tmp * ((b_a_tmp * cb_a_tmp_tmp * p_a_tmp_tmp *
    rb_a_tmp - cb_a_tmp * b_a_tmp_tmp * ab_a_tmp) + ob_a_tmp * sb_a_tmp) /
    m->contents) - bb_a * gain_el->contents * gamma_quadratic_dv->contents *
    sigma_5 * sigma_7 * ((K_p_T->contents * l_2->contents * cos(u_in[6] *
    gain_el->contents) - K_p_M->contents * cb_a_tmp_tmp * p_a_tmp_tmp) + y_a *
    p_a_tmp_tmp * t_a_tmp_tmp) / (a * a)) + qb_a_tmp * b_cost_tmp *
    gain_el->contents * gamma_quadratic_dv->contents * sigma_6 * sigma_7 *
    ((ab_a + x_a * cb_a_tmp_tmp * p_a_tmp_tmp) - K_p_T->contents * l_z->contents
     * sin(u_in[6] * gain_el->contents) * t_a_tmp_tmp) / (b_a * b_a)) + ac_a_tmp
    * gain_el->contents * gamma_quadratic_dv->contents * sigma_4 * sigma_7 *
    ((K_p_T->contents * l_z->contents * cos(u_in[6] * gain_el->contents) +
      K_p_M->contents * sin(u_in[6] * gain_el->contents) * t_a_tmp_tmp) + y_a *
     cb_a_tmp_tmp * p_a_tmp_tmp) / (c_a * c_a)) - f_a_tmp_tmp *
    gain_el->contents * gamma_quadratic_dv->contents * sigma_3 * sigma_7 *
    (((p_a_tmp * b_a_tmp_tmp * c_a_tmp_tmp - p_a_tmp_tmp * t_a_tmp_tmp *
       ab_a_tmp * c_a_tmp_tmp) - p_a_tmp * a_tmp_tmp * d_a_tmp_tmp * e_a_tmp_tmp)
     + cb_a_tmp_tmp * p_a_tmp_tmp * a_tmp_tmp * ab_a_tmp) / v_a;
  a = I_zz->contents;
  b_a = I_xx->contents;
  c_a = I_yy->contents;
  b_a_tmp = gc_a_tmp * gain_el->contents * sigma_7;
  cb_a_tmp = b_a_tmp * s_a_tmp;
  ob_a_tmp = b_a_tmp * q_a_tmp_tmp * u_a_tmp_tmp;
  qb_a_tmp = 2.0 * t_a_tmp;
  ab_a = t_a_tmp * c_cost_tmp;
  m_a_tmp_tmp = qb_a_tmp * cost_tmp;
  l_a_tmp_tmp = gc_a_tmp * d_cost_tmp;
  computed_gradient[7] = (((((mb_a_tmp * ((cb_a_tmp * ab_a_tmp * d_a_tmp_tmp +
    ob_a_tmp * bb_a_tmp) + b_a_tmp * h_a_tmp * db_a_tmp_tmp * q_a_tmp_tmp) /
    m->contents - ib_a_tmp_tmp * (desired_el_value->contents - k_a_tmp_tmp) /
    gain_el->contents) - nb_a_tmp * ((b_a_tmp * db_a_tmp_tmp * q_a_tmp_tmp *
    rb_a_tmp - cb_a_tmp * b_a_tmp_tmp * ab_a_tmp) + ob_a_tmp * sb_a_tmp) /
    m->contents) - m_a_tmp_tmp * gain_el->contents *
    gamma_quadratic_dv->contents * sigma_5 * sigma_7 * ((K_p_M->contents *
    db_a_tmp_tmp * q_a_tmp_tmp - K_p_T->contents * l_2->contents * cos(u_in[7] *
    gain_el->contents)) + y_a * q_a_tmp_tmp * u_a_tmp_tmp) / (a * a)) - qb_a_tmp
    * b_cost_tmp * gain_el->contents * gamma_quadratic_dv->contents * sigma_6 *
    sigma_7 * ((w_a + x_a * db_a_tmp_tmp * q_a_tmp_tmp) + K_p_T->contents *
               l_z->contents * sin(u_in[7] * gain_el->contents) * u_a_tmp_tmp) /
    (b_a * b_a)) + ab_a * gain_el->contents * gamma_quadratic_dv->contents *
    sigma_4 * sigma_7 * ((K_p_T->contents * l_z->contents * cos(u_in[7] *
    gain_el->contents) - K_p_M->contents * sin(u_in[7] * gain_el->contents) *
    u_a_tmp_tmp) + y_a * db_a_tmp_tmp * q_a_tmp_tmp) / (c_a * c_a)) -
    l_a_tmp_tmp * gain_el->contents * gamma_quadratic_dv->contents * sigma_3 *
    sigma_7 * (((s_a_tmp * b_a_tmp_tmp * c_a_tmp_tmp - q_a_tmp_tmp * u_a_tmp_tmp
                 * ab_a_tmp * c_a_tmp_tmp) - s_a_tmp * a_tmp_tmp * d_a_tmp_tmp *
                e_a_tmp_tmp) + db_a_tmp_tmp * q_a_tmp_tmp * a_tmp_tmp * ab_a_tmp)
    / v_a;
  a = I_yy->contents;
  b_a = I_zz->contents;
  c_a = I_xx->contents;
  ib_a_tmp_tmp = dc_a_tmp * gain_az->contents * sigma_7;
  b_a_tmp = ib_a_tmp_tmp * j_a_tmp;
  cb_a_tmp = b_a_tmp * ab_a_tmp_tmp;
  ob_a_tmp = 2.0 * b_cost_tmp_tmp * gamma_quadratic_du->contents;
  computed_gradient[8] = (((((nb_a_tmp * (cb_a_tmp * sb_a_tmp - b_a_tmp *
    r_a_tmp_tmp * rb_a_tmp) / m->contents - mb_a_tmp * (cb_a_tmp * bb_a_tmp -
    ib_a_tmp_tmp * h_a_tmp * j_a_tmp * r_a_tmp_tmp) / m->contents) - ob_a_tmp *
    (desired_az_value->contents - i_a_tmp) / gain_az->contents) - s_a *
    gain_az->contents * gamma_quadratic_dv->contents * sigma_4 * sigma_7 *
    j_a_tmp * (K_p_M->contents * cos(u_in[8] * gain_az->contents) + yb_a_tmp *
               r_a_tmp_tmp) / (a * a)) + t_a * gain_az->contents *
    gamma_quadratic_dv->contents * sigma_5 * sigma_7 * j_a_tmp *
    (K_p_M->contents * r_a_tmp_tmp - K_p_T->contents * l_4->contents * cos(u_in
    [8] * gain_az->contents)) / (b_a * b_a)) + ub_a_tmp * b_cost_tmp *
    gain_az->contents * gamma_quadratic_dv->contents * sigma_6 * sigma_7 *
    j_a_tmp * (l_z->contents * ab_a_tmp_tmp - l_1->contents * r_a_tmp_tmp) /
    (c_a * c_a)) - u_a * gain_az->contents * gamma_quadratic_dv->contents *
    sigma_3 * sigma_7 * sin(Phi->contents + i_a_tmp) * j_a_tmp * ab_a_tmp / v_a;
  a = I_yy->contents;
  b_a = I_zz->contents;
  c_a = I_xx->contents;
  ib_a_tmp_tmp = ec_a_tmp * gain_az->contents * sigma_7;
  b_a_tmp = ib_a_tmp_tmp * m_a_tmp;
  cb_a_tmp = b_a_tmp * bb_a_tmp_tmp;
  computed_gradient[9] = (((((nb_a_tmp * (cb_a_tmp * sb_a_tmp - b_a_tmp *
    s_a_tmp_tmp * rb_a_tmp) / m->contents - mb_a_tmp * (cb_a_tmp * bb_a_tmp -
    ib_a_tmp_tmp * h_a_tmp * m_a_tmp * s_a_tmp_tmp) / m->contents) - ob_a_tmp *
    (desired_az_value->contents - l_a_tmp) / gain_az->contents) + a_tmp *
    gain_az->contents * gamma_quadratic_dv->contents * sigma_4 * sigma_7 *
    m_a_tmp * (K_p_M->contents * cos(u_in[9] * gain_az->contents) - yb_a_tmp *
               s_a_tmp_tmp) / (a * a)) - g_a_tmp_tmp * gain_az->contents *
    gamma_quadratic_dv->contents * sigma_5 * sigma_7 * m_a_tmp *
    (K_p_M->contents * s_a_tmp_tmp + K_p_T->contents * l_4->contents * cos(u_in
    [9] * gain_az->contents)) / (b_a * b_a)) + vb_a_tmp * b_cost_tmp *
    gain_az->contents * gamma_quadratic_dv->contents * sigma_6 * sigma_7 *
    m_a_tmp * (l_z->contents * bb_a_tmp_tmp + l_1->contents * s_a_tmp_tmp) /
    (c_a * c_a)) - x * gain_az->contents * gamma_quadratic_dv->contents *
    sigma_3 * sigma_7 * sin(Phi->contents + l_a_tmp) * m_a_tmp * ab_a_tmp / v_a;
  a = I_yy->contents;
  b_a = I_zz->contents;
  c_a = I_xx->contents;
  ib_a_tmp_tmp = fc_a_tmp * gain_az->contents * sigma_7;
  b_a_tmp = ib_a_tmp_tmp * p_a_tmp;
  cb_a_tmp = b_a_tmp * cb_a_tmp_tmp;
  computed_gradient[10] = (((((nb_a_tmp * (cb_a_tmp * sb_a_tmp - b_a_tmp *
    t_a_tmp_tmp * rb_a_tmp) / m->contents - mb_a_tmp * (cb_a_tmp * bb_a_tmp -
    ib_a_tmp_tmp * h_a_tmp * p_a_tmp * t_a_tmp_tmp) / m->contents) - ob_a_tmp *
    (desired_az_value->contents - o_a_tmp) / gain_az->contents) - ac_a_tmp *
    gain_az->contents * gamma_quadratic_dv->contents * sigma_4 * sigma_7 *
    p_a_tmp * (K_p_M->contents * cos(u_in[10] * gain_az->contents) - y_a *
               t_a_tmp_tmp) / (a * a)) + bb_a * gain_az->contents *
    gamma_quadratic_dv->contents * sigma_5 * sigma_7 * p_a_tmp *
    (K_p_M->contents * t_a_tmp_tmp + K_p_T->contents * l_3->contents * cos(u_in
    [10] * gain_az->contents)) / (b_a * b_a)) + wb_a_tmp * b_cost_tmp *
    gain_az->contents * gamma_quadratic_dv->contents * sigma_6 * sigma_7 *
    p_a_tmp * (l_z->contents * cb_a_tmp_tmp + l_2->contents * t_a_tmp_tmp) /
    (c_a * c_a)) - f_a_tmp_tmp * gain_az->contents *
    gamma_quadratic_dv->contents * sigma_3 * sigma_7 * sin(Phi->contents +
    o_a_tmp) * p_a_tmp * ab_a_tmp / v_a;
  a = I_yy->contents;
  b_a = I_zz->contents;
  c_a = I_xx->contents;
  ib_a_tmp_tmp = gc_a_tmp * gain_az->contents * sigma_7;
  b_a_tmp = ib_a_tmp_tmp * s_a_tmp;
  cb_a_tmp = b_a_tmp * db_a_tmp_tmp;
  computed_gradient[11] = (((((nb_a_tmp * (cb_a_tmp * sb_a_tmp - b_a_tmp *
    u_a_tmp_tmp * rb_a_tmp) / m->contents - mb_a_tmp * (cb_a_tmp * bb_a_tmp -
    ib_a_tmp_tmp * h_a_tmp * s_a_tmp * u_a_tmp_tmp) / m->contents) - ob_a_tmp *
    (desired_az_value->contents - r_a_tmp) / gain_az->contents) + ab_a *
    gain_az->contents * gamma_quadratic_dv->contents * sigma_4 * sigma_7 *
    s_a_tmp * (K_p_M->contents * cos(u_in[11] * gain_az->contents) + y_a *
               u_a_tmp_tmp) / (a * a)) - m_a_tmp_tmp * gain_az->contents *
    gamma_quadratic_dv->contents * sigma_5 * sigma_7 * s_a_tmp *
    (K_p_M->contents * u_a_tmp_tmp - K_p_T->contents * l_3->contents * cos(u_in
    [11] * gain_az->contents)) / (b_a * b_a)) + xb_a_tmp * b_cost_tmp *
    gain_az->contents * gamma_quadratic_dv->contents * sigma_6 * sigma_7 *
    s_a_tmp * (l_z->contents * db_a_tmp_tmp - l_2->contents * u_a_tmp_tmp) /
    (c_a * c_a)) - l_a_tmp_tmp * gain_az->contents *
    gamma_quadratic_dv->contents * sigma_3 * sigma_7 * sin(Phi->contents +
    r_a_tmp) * s_a_tmp * ab_a_tmp / v_a;
}

/*
 * Arguments    : int n
 *                double a
 *                const double x[12]
 *                int ix0
 *                double y[72]
 *                int iy0
 * Return Type  : void
 */
static void c_xaxpy(int n, double a, const double x[12], int ix0, double y[72],
                    int iy0)
{
  int k;
  if (!(a == 0.0)) {
    int i;
    i = n - 1;
    for (k = 0; k <= i; k++) {
      int i1;
      i1 = (iy0 + k) - 1;
      y[i1] += a * x[(ix0 + k) - 1];
    }
  }
}

/*
 * Arguments    : int n
 *                const double x_data[]
 *                int ix0
 * Return Type  : double
 */
static double c_xnrm2(int n, const double x_data[], int ix0)
{
  double scale;
  double y;
  int k;
  int kend;
  y = 0.0;
  scale = 3.3121686421112381E-170;
  kend = (ix0 + n) - 1;
  for (k = ix0; k <= kend; k++) {
    double absxk;
    absxk = fabs(x_data[k - 1]);
    if (absxk > scale) {
      double t;
      t = scale / absxk;
      y = y * t * t + 1.0;
      scale = absxk;
    } else {
      double t;
      t = absxk / scale;
      y += t * t;
    }
  }

  return scale * sqrt(y);
}

/*
 * Arguments    : const double xCurrent[12]
 *                const int finiteLB[13]
 *                int mLB
 *                const double lb[12]
 *                const int finiteUB[13]
 *                int mUB
 *                const double ub[12]
 *                const double lambda[25]
 *                int iL0
 * Return Type  : double
 */
static double computeComplError(const double xCurrent[12], const int finiteLB[13],
  int mLB, const double lb[12], const int finiteUB[13], int mUB, const double
  ub[12], const double lambda[25], int iL0)
{
  double nlpComplError;
  int idx;
  nlpComplError = 0.0;
  if (mLB + mUB > 0) {
    double lbDelta;
    double lbLambda;
    int i;
    int ubOffset;
    ubOffset = (iL0 + mLB) - 1;
    for (idx = 0; idx < mLB; idx++) {
      i = finiteLB[idx];
      lbDelta = xCurrent[i - 1] - lb[i - 1];
      lbLambda = lambda[(iL0 + idx) - 1];
      nlpComplError = fmax(nlpComplError, fmin(fabs(lbDelta * lbLambda), fmin
        (fabs(lbDelta), lbLambda)));
    }

    for (idx = 0; idx < mUB; idx++) {
      i = finiteUB[idx];
      lbDelta = ub[i - 1] - xCurrent[i - 1];
      lbLambda = lambda[ubOffset + idx];
      nlpComplError = fmax(nlpComplError, fmin(fabs(lbDelta * lbLambda), fmin
        (fabs(lbDelta), lbLambda)));
    }
  }

  return nlpComplError;
}

/*
 * Arguments    : const struct_T *obj
 *                double workspace[325]
 *                const double H[144]
 *                const double f[13]
 *                const double x[13]
 * Return Type  : double
 */
static double computeFval(const struct_T *obj, double workspace[325], const
  double H[144], const double f[13], const double x[13])
{
  double val;
  int idx;
  int k;
  switch (obj->objtype) {
   case 5:
    val = obj->gammaScalar * x[obj->nvar - 1];
    break;

   case 3:
    {
      linearForm_(obj->hasLinear, obj->nvar, workspace, H, f, x);
      val = 0.0;
      if (obj->nvar >= 1) {
        int ixlast;
        ixlast = obj->nvar;
        for (k = 0; k < ixlast; k++) {
          val += x[k] * workspace[k];
        }
      }
    }
    break;

   default:
    {
      int ixlast;
      linearForm_(obj->hasLinear, obj->nvar, workspace, H, f, x);
      ixlast = obj->nvar + 1;
      if (13 - ixlast < 400) {
        for (idx = ixlast; idx < 13; idx++) {
          workspace[idx - 1] = 0.5 * obj->beta * x[idx - 1] + obj->rho;
        }
      } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

        for (idx = ixlast; idx < 13; idx++) {
          workspace[idx - 1] = 0.5 * obj->beta * x[idx - 1] + obj->rho;
        }
      }

      val = 0.0;
      for (k = 0; k < 12; k++) {
        val += x[k] * workspace[k];
      }
    }
    break;
  }

  return val;
}

/*
 * Arguments    : const struct_T *obj
 *                double workspace[325]
 *                const double f[13]
 *                const double x[13]
 * Return Type  : double
 */
static double computeFval_ReuseHx(const struct_T *obj, double workspace[325],
  const double f[13], const double x[13])
{
  double val;
  int i;
  int k;
  switch (obj->objtype) {
   case 5:
    val = obj->gammaScalar * x[obj->nvar - 1];
    break;

   case 3:
    {
      if (obj->hasLinear) {
        int ixlast;
        ixlast = obj->nvar;
        if (ixlast < 400) {
          for (i = 0; i < ixlast; i++) {
            workspace[i] = 0.5 * obj->Hx[i] + f[i];
          }
        } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

          for (i = 0; i < ixlast; i++) {
            workspace[i] = 0.5 * obj->Hx[i] + f[i];
          }
        }

        val = 0.0;
        if (obj->nvar >= 1) {
          ixlast = obj->nvar;
          for (k = 0; k < ixlast; k++) {
            val += x[k] * workspace[k];
          }
        }
      } else {
        val = 0.0;
        if (obj->nvar >= 1) {
          int ixlast;
          ixlast = obj->nvar;
          for (k = 0; k < ixlast; k++) {
            val += x[k] * obj->Hx[k];
          }
        }

        val *= 0.5;
      }
    }
    break;

   default:
    {
      if (obj->hasLinear) {
        int ixlast;
        ixlast = obj->nvar;
        if (ixlast < 400) {
          if (ixlast - 1 >= 0) {
            memcpy(&workspace[0], &f[0], ixlast * sizeof(double));
          }
        } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

          for (i = 0; i < ixlast; i++) {
            workspace[i] = f[i];
          }
        }

        ixlast = 11 - obj->nvar;
        for (k = 0; k <= ixlast; k++) {
          workspace[obj->nvar + k] = obj->rho;
        }

        val = 0.0;
        for (k = 0; k < 12; k++) {
          workspace[k] += 0.5 * obj->Hx[k];
          val += x[k] * workspace[k];
        }
      } else {
        int ixlast;
        val = 0.0;
        for (k = 0; k < 12; k++) {
          val += x[k] * obj->Hx[k];
        }

        val *= 0.5;
        ixlast = obj->nvar + 1;
        for (k = ixlast; k < 13; k++) {
          val += x[k - 1] * obj->rho;
        }
      }
    }
    break;
  }

  return val;
}

/*
 * Arguments    : double workspace[13]
 *                int nVar
 *                const double grad[13]
 *                const int finiteFixed[13]
 *                int mFixed
 *                const int finiteLB[13]
 *                int mLB
 *                const int finiteUB[13]
 *                int mUB
 *                const double lambda[25]
 * Return Type  : void
 */
static void computeGradLag(double workspace[13], int nVar, const double grad[13],
  const int finiteFixed[13], int mFixed, const int finiteLB[13], int mLB, const
  int finiteUB[13], int mUB, const double lambda[25])
{
  int b_i;
  int i;
  int iL0;
  int idx;
  if (nVar < 400) {
    if (nVar - 1 >= 0) {
      memcpy(&workspace[0], &grad[0], nVar * sizeof(double));
    }
  } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

    for (i = 0; i < nVar; i++) {
      workspace[i] = grad[i];
    }
  }

  for (idx = 0; idx < mFixed; idx++) {
    b_i = finiteFixed[idx];
    workspace[b_i - 1] += lambda[idx];
  }

  for (idx = 0; idx < mLB; idx++) {
    b_i = finiteLB[idx];
    workspace[b_i - 1] -= lambda[mFixed + idx];
  }

  iL0 = mFixed + mLB;
  for (idx = 0; idx < mUB; idx++) {
    b_i = finiteUB[idx];
    workspace[b_i - 1] += lambda[iL0 + idx];
  }
}

/*
 * Arguments    : struct_T *obj
 *                const double H[144]
 *                const double f[13]
 *                const double x[13]
 * Return Type  : void
 */
static void computeGrad_StoreHx(struct_T *obj, const double H[144], const double
  f[13], const double x[13])
{
  int ia;
  int iac;
  int iy;
  int k;
  switch (obj->objtype) {
   case 5:
    {
      int i;
      i = obj->nvar;
      if (i - 1 < 400) {
        if (i - 2 >= 0) {
          memset(&obj->grad[0], 0, (i - 1) * sizeof(double));
        }
      } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

        for (iy = 0; iy <= i - 2; iy++) {
          obj->grad[iy] = 0.0;
        }
      }

      obj->grad[obj->nvar - 1] = obj->gammaScalar;
    }
    break;

   case 3:
    {
      int i;
      int ix;
      int lda;
      int m_tmp_tmp;
      m_tmp_tmp = obj->nvar - 1;
      lda = obj->nvar;
      if (obj->nvar != 0) {
        if (m_tmp_tmp + 1 < 400) {
          if (m_tmp_tmp >= 0) {
            memset(&obj->Hx[0], 0, (m_tmp_tmp + 1) * sizeof(double));
          }
        } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

          for (iy = 0; iy <= m_tmp_tmp; iy++) {
            obj->Hx[iy] = 0.0;
          }
        }

        ix = 0;
        i = obj->nvar * (obj->nvar - 1) + 1;
        for (iac = 1; lda < 0 ? iac >= i : iac <= i; iac += lda) {
          int i1;
          i1 = iac + m_tmp_tmp;
          for (ia = iac; ia <= i1; ia++) {
            int i2;
            i2 = ia - iac;
            obj->Hx[i2] += H[ia - 1] * x[ix];
          }

          ix++;
        }
      }

      i = obj->nvar;
      if (i - 1 >= 0) {
        memcpy(&obj->grad[0], &obj->Hx[0], i * sizeof(double));
      }

      if (obj->hasLinear && (obj->nvar >= 1)) {
        ix = obj->nvar - 1;
        if (ix + 1 < 400) {
          for (k = 0; k <= ix; k++) {
            obj->grad[k] += f[k];
          }
        } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

          for (k = 0; k <= ix; k++) {
            obj->grad[k] += f[k];
          }
        }
      }
    }
    break;

   default:
    {
      int i;
      int i1;
      int ix;
      int lda;
      int m_tmp_tmp;
      m_tmp_tmp = obj->nvar - 1;
      lda = obj->nvar;
      if (obj->nvar != 0) {
        if (m_tmp_tmp + 1 < 400) {
          if (m_tmp_tmp >= 0) {
            memset(&obj->Hx[0], 0, (m_tmp_tmp + 1) * sizeof(double));
          }
        } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

          for (iy = 0; iy <= m_tmp_tmp; iy++) {
            obj->Hx[iy] = 0.0;
          }
        }

        ix = 0;
        i = obj->nvar * (obj->nvar - 1) + 1;
        for (iac = 1; lda < 0 ? iac >= i : iac <= i; iac += lda) {
          i1 = iac + m_tmp_tmp;
          for (ia = iac; ia <= i1; ia++) {
            int i2;
            i2 = ia - iac;
            obj->Hx[i2] += H[ia - 1] * x[ix];
          }

          ix++;
        }
      }

      i = obj->nvar + 1;
      if (13 - i < 400) {
        for (iy = i; iy < 13; iy++) {
          obj->Hx[iy - 1] = obj->beta * x[iy - 1];
        }
      } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

        for (iy = i; iy < 13; iy++) {
          obj->Hx[iy - 1] = obj->beta * x[iy - 1];
        }
      }

      memcpy(&obj->grad[0], &obj->Hx[0], 12U * sizeof(double));
      if (obj->hasLinear && (obj->nvar >= 1)) {
        ix = obj->nvar - 1;
        if (ix + 1 < 400) {
          for (k = 0; k <= ix; k++) {
            obj->grad[k] += f[k];
          }
        } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

          for (k = 0; k <= ix; k++) {
            obj->grad[k] += f[k];
          }
        }
      }

      if (12 - obj->nvar >= 1) {
        ix = obj->nvar;
        i = 11 - obj->nvar;
        if (i + 1 < 400) {
          for (k = 0; k <= i; k++) {
            i1 = ix + k;
            obj->grad[i1] += obj->rho;
          }
        } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4) \
 private(iy)

          for (k = 0; k <= i; k++) {
            iy = ix + k;
            obj->grad[iy] += obj->rho;
          }
        }
      }
    }
    break;
  }
}

/*
 * Arguments    : f_struct_T *obj
 *                int nrows
 * Return Type  : void
 */
static void computeQ_(f_struct_T *obj, int nrows)
{
  double work[25];
  int b_i;
  int i;
  int iQR0;
  int ia;
  int idx;
  int m;
  int n;
  i = obj->minRowCol;
  for (idx = 0; idx < i; idx++) {
    iQR0 = 25 * idx + idx;
    n = obj->mrows - idx;
    if (n - 2 >= 0) {
      memcpy(&obj->Q[iQR0 + 1], &obj->QR[iQR0 + 1], (((n + iQR0) - iQR0) - 1) *
             sizeof(double));
    }
  }

  m = obj->mrows;
  n = obj->minRowCol;
  if (nrows >= 1) {
    int i1;
    int itau;
    i = nrows - 1;
    for (idx = n; idx <= i; idx++) {
      ia = idx * 25;
      i1 = m - 1;
      memset(&obj->Q[ia], 0, (((i1 + ia) - ia) + 1) * sizeof(double));
      obj->Q[ia + idx] = 1.0;
    }

    itau = obj->minRowCol - 1;
    memset(&work[0], 0, 25U * sizeof(double));
    for (b_i = obj->minRowCol; b_i >= 1; b_i--) {
      int iaii;
      iaii = b_i + (b_i - 1) * 25;
      if (b_i < nrows) {
        int lastc;
        int lastv;
        obj->Q[iaii - 1] = 1.0;
        idx = iaii + 25;
        if (obj->tau[itau] != 0.0) {
          bool exitg2;
          lastv = m - b_i;
          iQR0 = (iaii + m) - b_i;
          while ((lastv + 1 > 0) && (obj->Q[iQR0 - 1] == 0.0)) {
            lastv--;
            iQR0--;
          }

          lastc = (nrows - b_i) - 1;
          exitg2 = false;
          while ((!exitg2) && (lastc + 1 > 0)) {
            int exitg1;
            iQR0 = (iaii + lastc * 25) + 25;
            ia = iQR0;
            do {
              exitg1 = 0;
              if (ia <= iQR0 + lastv) {
                if (obj->Q[ia - 1] != 0.0) {
                  exitg1 = 1;
                } else {
                  ia++;
                }
              } else {
                lastc--;
                exitg1 = 2;
              }
            } while (exitg1 == 0);

            if (exitg1 == 1) {
              exitg2 = true;
            }
          }
        } else {
          lastv = -1;
          lastc = -1;
        }

        if (lastv + 1 > 0) {
          double c;
          if (lastc + 1 != 0) {
            if (lastc >= 0) {
              memset(&work[0], 0, (lastc + 1) * sizeof(double));
            }

            i = (iaii + 25 * lastc) + 25;
            for (n = idx; n <= i; n += 25) {
              c = 0.0;
              i1 = n + lastv;
              for (ia = n; ia <= i1; ia++) {
                c += obj->Q[ia - 1] * obj->Q[((iaii + ia) - n) - 1];
              }

              iQR0 = div_nde_s32_floor((n - iaii) - 25, 25);
              work[iQR0] += c;
            }
          }

          if (!(-obj->tau[itau] == 0.0)) {
            iQR0 = iaii;
            for (idx = 0; idx <= lastc; idx++) {
              c = work[idx];
              if (c != 0.0) {
                c *= -obj->tau[itau];
                i = iQR0 + 25;
                i1 = lastv + iQR0;
                for (n = i; n <= i1 + 25; n++) {
                  obj->Q[n - 1] += obj->Q[((iaii + n) - iQR0) - 26] * c;
                }
              }

              iQR0 += 25;
            }
          }
        }
      }

      if (b_i < m) {
        iQR0 = iaii + 1;
        i = (iaii + m) - b_i;
        for (n = iQR0; n <= i; n++) {
          obj->Q[n - 1] *= -obj->tau[itau];
        }
      }

      obj->Q[iaii - 1] = 1.0 - obj->tau[itau];
      for (idx = 0; idx <= b_i - 2; idx++) {
        obj->Q[(iaii - idx) - 2] = 0.0;
      }

      itau--;
    }
  }
}

/*
 * Arguments    : const double H[144]
 *                i_struct_T *solution
 *                e_struct_T *memspace
 *                const f_struct_T *qrmanager
 *                g_struct_T *cholmanager
 *                const struct_T *objective
 *                bool alwaysPositiveDef
 * Return Type  : void
 */
static void compute_deltax(const double H[144], i_struct_T *solution, e_struct_T
  *memspace, const f_struct_T *qrmanager, g_struct_T *cholmanager, const
  struct_T *objective, bool alwaysPositiveDef)
{
  int b_idx;
  int i1;
  int idx;
  int idx_row;
  int k;
  int mNull;
  int nVar;
  nVar = qrmanager->mrows - 1;
  mNull = qrmanager->mrows - qrmanager->ncols;
  if (mNull <= 0) {
    if (nVar + 1 < 400) {
      if (nVar >= 0) {
        memset(&solution->searchDir[0], 0, (nVar + 1) * sizeof(double));
      }
    } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

      for (idx = 0; idx <= nVar; idx++) {
        solution->searchDir[idx] = 0.0;
      }
    }
  } else {
    if (nVar + 1 < 400) {
      for (idx = 0; idx <= nVar; idx++) {
        solution->searchDir[idx] = -objective->grad[idx];
      }
    } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

      for (idx = 0; idx <= nVar; idx++) {
        solution->searchDir[idx] = -objective->grad[idx];
      }
    }

    if (qrmanager->ncols <= 0) {
      switch (objective->objtype) {
       case 5:
        break;

       case 3:
        {
          double smax;
          int b_mNull;
          int ix;
          int jjA;
          int nVars;
          if (alwaysPositiveDef) {
            cholmanager->ndims = qrmanager->mrows;
            for (b_idx = 0; b_idx <= nVar; b_idx++) {
              ix = (nVar + 1) * b_idx;
              jjA = 25 * b_idx;
              for (k = 0; k <= nVar; k++) {
                cholmanager->FMat[jjA + k] = H[ix + k];
              }
            }

            cholmanager->info = xpotrf(qrmanager->mrows, cholmanager->FMat);
          } else {
            b_mNull = qrmanager->mrows - 1;
            cholmanager->ndims = qrmanager->mrows;
            for (b_idx = 0; b_idx <= b_mNull; b_idx++) {
              ix = qrmanager->mrows * b_idx;
              jjA = 25 * b_idx;
              for (k = 0; k <= b_mNull; k++) {
                cholmanager->FMat[jjA + k] = H[ix + k];
              }
            }

            if (qrmanager->mrows < 1) {
              nVars = -1;
            } else {
              nVars = 0;
              if (qrmanager->mrows > 1) {
                smax = fabs(cholmanager->FMat[0]);
                for (k = 2; k <= b_mNull + 1; k++) {
                  double s;
                  s = fabs(cholmanager->FMat[(k - 1) * 26]);
                  if (s > smax) {
                    nVars = k - 1;
                    smax = s;
                  }
                }
              }
            }

            cholmanager->regTol_ = fmax(fabs(cholmanager->FMat[nVars + 25 *
              nVars]) * 2.2204460492503131E-16, 0.0);
            fullColLDL2_(cholmanager, qrmanager->mrows);
            if (cholmanager->ConvexCheck) {
              b_idx = 0;
              int exitg1;
              do {
                exitg1 = 0;
                if (b_idx <= b_mNull) {
                  if (cholmanager->FMat[b_idx + 25 * b_idx] <= 0.0) {
                    cholmanager->info = -b_idx - 1;
                    exitg1 = 1;
                  } else {
                    b_idx++;
                  }
                } else {
                  cholmanager->ConvexCheck = false;
                  exitg1 = 1;
                }
              } while (exitg1 == 0);
            }
          }

          if (cholmanager->info != 0) {
            solution->state = -6;
          } else if (alwaysPositiveDef) {
            solve(cholmanager, solution->searchDir);
          } else {
            int i;
            b_mNull = cholmanager->ndims - 2;
            if (cholmanager->ndims != 0) {
              for (k = 0; k <= b_mNull + 1; k++) {
                jjA = k + k * 25;
                i = b_mNull - k;
                for (b_idx = 0; b_idx <= i; b_idx++) {
                  ix = (k + b_idx) + 1;
                  solution->searchDir[ix] -= solution->searchDir[k] *
                    cholmanager->FMat[(jjA + b_idx) + 1];
                }
              }
            }

            i = cholmanager->ndims;
            for (b_idx = 0; b_idx < i; b_idx++) {
              solution->searchDir[b_idx] /= cholmanager->FMat[b_idx + 25 * b_idx];
            }

            b_mNull = cholmanager->ndims;
            if (cholmanager->ndims != 0) {
              for (k = b_mNull; k >= 1; k--) {
                nVars = (k - 1) * 25;
                smax = solution->searchDir[k - 1];
                i = k + 1;
                for (b_idx = b_mNull; b_idx >= i; b_idx--) {
                  smax -= cholmanager->FMat[(nVars + b_idx) - 1] *
                    solution->searchDir[b_idx - 1];
                }

                solution->searchDir[k - 1] = smax;
              }
            }
          }
        }
        break;

       default:
        {
          if (alwaysPositiveDef) {
            int ix;
            int nVars;
            nVars = objective->nvar;
            cholmanager->ndims = objective->nvar;
            for (b_idx = 0; b_idx < nVars; b_idx++) {
              int jjA;
              ix = nVars * b_idx;
              jjA = 25 * b_idx;
              for (k = 0; k < nVars; k++) {
                cholmanager->FMat[jjA + k] = H[ix + k];
              }
            }

            cholmanager->info = xpotrf(objective->nvar, cholmanager->FMat);
            if (cholmanager->info != 0) {
              solution->state = -6;
            } else {
              double smax;
              int i;
              solve(cholmanager, solution->searchDir);
              smax = 1.0 / objective->beta;
              ix = objective->nvar + 1;
              i = qrmanager->mrows;
              if ((i - ix) + 1 < 400) {
                for (idx = ix; idx <= i; idx++) {
                  solution->searchDir[idx - 1] *= smax;
                }
              } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

                for (idx = ix; idx <= i; idx++) {
                  solution->searchDir[idx - 1] *= smax;
                }
              }
            }
          }
        }
        break;
      }
    } else {
      int nullStartIdx;
      nullStartIdx = 25 * qrmanager->ncols + 1;
      if (objective->objtype == 5) {
        int jjA;
        for (b_idx = 0; b_idx < mNull; b_idx++) {
          memspace->workspace_double[b_idx] = -qrmanager->Q[nVar + 25 *
            (qrmanager->ncols + b_idx)];
        }

        jjA = qrmanager->mrows - 1;
        if (qrmanager->mrows != 0) {
          int i;
          int ix;
          if (jjA + 1 < 400) {
            if (jjA >= 0) {
              memset(&solution->searchDir[0], 0, (jjA + 1) * sizeof(double));
            }
          } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

            for (idx = 0; idx <= jjA; idx++) {
              solution->searchDir[idx] = 0.0;
            }
          }

          ix = 0;
          i = nullStartIdx + 25 * (mNull - 1);
          for (k = nullStartIdx; k <= i; k += 25) {
            int b_mNull;
            b_mNull = k + jjA;
            for (b_idx = k; b_idx <= b_mNull; b_idx++) {
              int nVars;
              nVars = b_idx - k;
              solution->searchDir[nVars] += qrmanager->Q[b_idx - 1] *
                memspace->workspace_double[ix];
            }

            ix++;
          }
        }
      } else {
        double smax;
        int b_mNull;
        int i;
        int nVars;
        if (objective->objtype == 3) {
          i = qrmanager->mrows - qrmanager->ncols;
          xgemm(qrmanager->mrows, i, qrmanager->mrows, H, qrmanager->mrows,
                qrmanager->Q, nullStartIdx, memspace->workspace_double);
          b_xgemm(i, i, qrmanager->mrows, qrmanager->Q, nullStartIdx,
                  memspace->workspace_double, cholmanager->FMat);
        } else if (alwaysPositiveDef) {
          nVars = qrmanager->mrows;
          b_mNull = qrmanager->mrows - qrmanager->ncols;
          xgemm(objective->nvar, b_mNull, objective->nvar, H, objective->nvar,
                qrmanager->Q, nullStartIdx, memspace->workspace_double);

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4) \
 private(idx_row,i1)

          for (idx = 0; idx < b_mNull; idx++) {
            i1 = objective->nvar + 1;
            for (idx_row = i1; idx_row <= nVars; idx_row++) {
              memspace->workspace_double[(idx_row + 25 * idx) - 1] =
                objective->beta * qrmanager->Q[(idx_row + 25 * (idx +
                qrmanager->ncols)) - 1];
            }
          }

          b_xgemm(b_mNull, b_mNull, qrmanager->mrows, qrmanager->Q, nullStartIdx,
                  memspace->workspace_double, cholmanager->FMat);
        }

        if (alwaysPositiveDef) {
          cholmanager->ndims = mNull;
          cholmanager->info = xpotrf(mNull, cholmanager->FMat);
        } else {
          cholmanager->ndims = mNull;
          nVars = 0;
          if (mNull > 1) {
            smax = fabs(cholmanager->FMat[0]);
            for (k = 2; k <= mNull; k++) {
              double s;
              s = fabs(cholmanager->FMat[(k - 1) * 26]);
              if (s > smax) {
                nVars = k - 1;
                smax = s;
              }
            }
          }

          cholmanager->regTol_ = fmax(fabs(cholmanager->FMat[nVars + 25 * nVars])
            * 2.2204460492503131E-16, 0.0);
          fullColLDL2_(cholmanager, mNull);
          if (cholmanager->ConvexCheck) {
            b_idx = 0;
            int exitg1;
            do {
              exitg1 = 0;
              if (b_idx <= mNull - 1) {
                if (cholmanager->FMat[b_idx + 25 * b_idx] <= 0.0) {
                  cholmanager->info = -b_idx - 1;
                  exitg1 = 1;
                } else {
                  b_idx++;
                }
              } else {
                cholmanager->ConvexCheck = false;
                exitg1 = 1;
              }
            } while (exitg1 == 0);
          }
        }

        if (cholmanager->info != 0) {
          solution->state = -6;
        } else {
          int ix;
          int jjA;
          if (qrmanager->mrows != 0) {
            if (mNull < 400) {
              memset(&memspace->workspace_double[0], 0, mNull * sizeof(double));
            } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

              for (idx = 0; idx < mNull; idx++) {
                memspace->workspace_double[idx] = 0.0;
              }
            }

            i = nullStartIdx + 25 * (mNull - 1);
            for (k = nullStartIdx; k <= i; k += 25) {
              smax = 0.0;
              b_mNull = k + nVar;
              for (b_idx = k; b_idx <= b_mNull; b_idx++) {
                smax += qrmanager->Q[b_idx - 1] * objective->grad[b_idx - k];
              }

              b_mNull = div_nde_s32_floor(k - nullStartIdx, 25);
              memspace->workspace_double[b_mNull] += -smax;
            }
          }

          if (alwaysPositiveDef) {
            b_mNull = cholmanager->ndims;
            if (cholmanager->ndims != 0) {
              for (k = 0; k < b_mNull; k++) {
                nVars = k * 25;
                smax = memspace->workspace_double[k];
                for (b_idx = 0; b_idx < k; b_idx++) {
                  smax -= cholmanager->FMat[nVars + b_idx] *
                    memspace->workspace_double[b_idx];
                }

                memspace->workspace_double[k] = smax / cholmanager->FMat[nVars +
                  k];
              }
            }

            b_mNull = cholmanager->ndims;
            if (cholmanager->ndims != 0) {
              for (k = b_mNull; k >= 1; k--) {
                jjA = (k + (k - 1) * 25) - 1;
                memspace->workspace_double[k - 1] /= cholmanager->FMat[jjA];
                for (b_idx = 0; b_idx <= k - 2; b_idx++) {
                  ix = (k - b_idx) - 2;
                  memspace->workspace_double[ix] -= memspace->workspace_double[k
                    - 1] * cholmanager->FMat[(jjA - b_idx) - 1];
                }
              }
            }
          } else {
            b_mNull = cholmanager->ndims - 2;
            if (cholmanager->ndims != 0) {
              for (k = 0; k <= b_mNull + 1; k++) {
                jjA = k + k * 25;
                i = b_mNull - k;
                for (b_idx = 0; b_idx <= i; b_idx++) {
                  ix = (k + b_idx) + 1;
                  memspace->workspace_double[ix] -= memspace->workspace_double[k]
                    * cholmanager->FMat[(jjA + b_idx) + 1];
                }
              }
            }

            i = cholmanager->ndims;
            for (b_idx = 0; b_idx < i; b_idx++) {
              memspace->workspace_double[b_idx] /= cholmanager->FMat[b_idx + 25 *
                b_idx];
            }

            b_mNull = cholmanager->ndims;
            if (cholmanager->ndims != 0) {
              for (k = b_mNull; k >= 1; k--) {
                nVars = (k - 1) * 25;
                smax = memspace->workspace_double[k - 1];
                i = k + 1;
                for (b_idx = b_mNull; b_idx >= i; b_idx--) {
                  smax -= cholmanager->FMat[(nVars + b_idx) - 1] *
                    memspace->workspace_double[b_idx - 1];
                }

                memspace->workspace_double[k - 1] = smax;
              }
            }
          }

          jjA = qrmanager->mrows - 1;
          if (qrmanager->mrows != 0) {
            if (jjA + 1 < 400) {
              if (jjA >= 0) {
                memset(&solution->searchDir[0], 0, (jjA + 1) * sizeof(double));
              }
            } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

              for (idx = 0; idx <= jjA; idx++) {
                solution->searchDir[idx] = 0.0;
              }
            }

            ix = 0;
            i = nullStartIdx + 25 * (mNull - 1);
            for (k = nullStartIdx; k <= i; k += 25) {
              b_mNull = k + jjA;
              for (b_idx = k; b_idx <= b_mNull; b_idx++) {
                nVars = b_idx - k;
                solution->searchDir[nVars] += qrmanager->Q[b_idx - 1] *
                  memspace->workspace_double[ix];
              }

              ix++;
            }
          }
        }
      }
    }
  }
}

/*
 * Arguments    : int x[25]
 *                int xLen
 *                int workspace[25]
 *                int xMin
 *                int xMax
 * Return Type  : void
 */
static void countsort(int x[25], int xLen, int workspace[25], int xMin, int xMax)
{
  int idx;
  int idxEnd;
  int idxFill;
  int idxW;
  if ((xLen > 1) && (xMax > xMin)) {
    int idxStart;
    int maxOffset;
    idxStart = xMax - xMin;
    if (idxStart + 1 < 400) {
      if (idxStart >= 0) {
        memset(&workspace[0], 0, (idxStart + 1) * sizeof(int));
      }
    } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

      for (idx = 0; idx <= idxStart; idx++) {
        workspace[idx] = 0;
      }
    }

    maxOffset = (xMax - xMin) - 1;
    for (idxEnd = 0; idxEnd < xLen; idxEnd++) {
      idxStart = x[idxEnd] - xMin;
      workspace[idxStart]++;
    }

    for (idxEnd = 2; idxEnd <= maxOffset + 2; idxEnd++) {
      workspace[idxEnd - 1] += workspace[idxEnd - 2];
    }

    idxStart = 1;
    idxEnd = workspace[0];
    for (idxW = 0; idxW <= maxOffset; idxW++) {
      for (idxFill = idxStart; idxFill <= idxEnd; idxFill++) {
        x[idxFill - 1] = idxW + xMin;
      }

      idxStart = workspace[idxW] + 1;
      idxEnd = workspace[idxW + 1];
    }

    if ((idxEnd - idxStart) + 1 < 400) {
      for (idx = idxStart; idx <= idxEnd; idx++) {
        x[idx - 1] = xMax;
      }
    } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

      for (idx = idxStart; idx <= idxEnd; idx++) {
        x[idx - 1] = xMax;
      }
    }
  }
}

/*
 * Arguments    : int n
 *                double a
 *                int ix0
 *                double y[36]
 *                int iy0
 * Return Type  : void
 */
static void d_xaxpy(int n, double a, int ix0, double y[36], int iy0)
{
  int k;
  if (!(a == 0.0)) {
    int i;
    i = n - 1;
    for (k = 0; k <= i; k++) {
      int i1;
      i1 = (iy0 + k) - 1;
      y[i1] += a * y[(ix0 + k) - 1];
    }
  }
}

/*
 * Arguments    : int n
 *                const double x[72]
 *                int ix0
 * Return Type  : double
 */
static double d_xnrm2(int n, const double x[72], int ix0)
{
  double scale;
  double y;
  int k;
  int kend;
  y = 0.0;
  scale = 3.3121686421112381E-170;
  kend = (ix0 + n) - 1;
  for (k = ix0; k <= kend; k++) {
    double absxk;
    absxk = fabs(x[k - 1]);
    if (absxk > scale) {
      double t;
      t = scale / absxk;
      y = y * t * t + 1.0;
      scale = absxk;
    } else {
      double t;
      t = absxk / scale;
      y += t * t;
    }
  }

  return scale * sqrt(y);
}

/*
 * Arguments    : f_struct_T *obj
 *                int idx
 * Return Type  : void
 */
static void deleteColMoveEnd(f_struct_T *obj, int idx)
{
  double c;
  double s;
  double temp_tmp;
  int b_k;
  int i;
  int k;
  if (obj->usedPivoting) {
    i = 1;
    while ((i <= obj->ncols) && (obj->jpvt[i - 1] != idx)) {
      i++;
    }

    idx = i;
  }

  if (idx >= obj->ncols) {
    obj->ncols--;
  } else {
    int b_i;
    int u0;
    obj->jpvt[idx - 1] = obj->jpvt[obj->ncols - 1];
    b_i = obj->minRowCol;
    for (k = 0; k < b_i; k++) {
      obj->QR[k + 25 * (idx - 1)] = obj->QR[k + 25 * (obj->ncols - 1)];
    }

    obj->ncols--;
    u0 = obj->mrows;
    i = obj->ncols;
    if (u0 <= i) {
      i = u0;
    }

    obj->minRowCol = i;
    if (idx < obj->mrows) {
      double c_temp_tmp;
      int QRk0;
      int b_temp_tmp;
      int endIdx;
      int n;
      u0 = obj->mrows - 1;
      endIdx = obj->ncols;
      if (u0 <= endIdx) {
        endIdx = u0;
      }

      k = endIdx;
      i = 25 * (idx - 1);
      while (k >= idx) {
        b_i = k + i;
        temp_tmp = obj->QR[b_i];
        xrotg(&obj->QR[(k + i) - 1], &temp_tmp, &c, &s);
        obj->QR[b_i] = temp_tmp;
        b_i = 25 * (k - 1);
        obj->QR[k + b_i] = 0.0;
        QRk0 = k + 25 * idx;
        n = obj->ncols - idx;
        if (n >= 1) {
          for (b_k = 0; b_k < n; b_k++) {
            b_temp_tmp = QRk0 + b_k * 25;
            temp_tmp = obj->QR[b_temp_tmp];
            c_temp_tmp = obj->QR[b_temp_tmp - 1];
            obj->QR[b_temp_tmp] = c * temp_tmp - s * c_temp_tmp;
            obj->QR[b_temp_tmp - 1] = c * c_temp_tmp + s * temp_tmp;
          }
        }

        n = obj->mrows;
        for (b_k = 0; b_k < n; b_k++) {
          b_temp_tmp = b_i + b_k;
          temp_tmp = obj->Q[b_temp_tmp + 25];
          c_temp_tmp = obj->Q[b_temp_tmp];
          obj->Q[b_temp_tmp + 25] = c * temp_tmp - s * c_temp_tmp;
          obj->Q[b_temp_tmp] = c * c_temp_tmp + s * temp_tmp;
        }

        k--;
      }

      b_i = idx + 1;
      for (k = b_i; k <= endIdx; k++) {
        u0 = 25 * (k - 1);
        i = k + u0;
        temp_tmp = obj->QR[i];
        xrotg(&obj->QR[(k + u0) - 1], &temp_tmp, &c, &s);
        obj->QR[i] = temp_tmp;
        QRk0 = k * 26;
        n = obj->ncols - k;
        if (n >= 1) {
          for (b_k = 0; b_k < n; b_k++) {
            b_temp_tmp = QRk0 + b_k * 25;
            temp_tmp = obj->QR[b_temp_tmp];
            c_temp_tmp = obj->QR[b_temp_tmp - 1];
            obj->QR[b_temp_tmp] = c * temp_tmp - s * c_temp_tmp;
            obj->QR[b_temp_tmp - 1] = c * c_temp_tmp + s * temp_tmp;
          }
        }

        n = obj->mrows;
        for (b_k = 0; b_k < n; b_k++) {
          b_temp_tmp = u0 + b_k;
          temp_tmp = obj->Q[b_temp_tmp + 25];
          c_temp_tmp = obj->Q[b_temp_tmp];
          obj->Q[b_temp_tmp + 25] = c * temp_tmp - s * c_temp_tmp;
          obj->Q[b_temp_tmp] = c * c_temp_tmp + s * temp_tmp;
        }
      }
    }
  }
}

/*
 * Arguments    : int numerator
 *                int denominator
 * Return Type  : int
 */
static int div_nde_s32_floor(int numerator, int denominator)
{
  int b_numerator;
  if (((numerator < 0) != (denominator < 0)) && (numerator % denominator != 0))
  {
    b_numerator = -1;
  } else {
    b_numerator = 0;
  }

  return numerator / denominator + b_numerator;
}

/*
 * Arguments    : const double H[144]
 *                const double f[13]
 *                i_struct_T *solution
 *                e_struct_T *memspace
 *                k_struct_T *workingset
 *                f_struct_T *qrmanager
 *                g_struct_T *cholmanager
 *                struct_T *objective
 *                h_struct_T *options
 *                int runTimeOptions_MaxIterations
 * Return Type  : void
 */
static void driver(const double H[144], const double f[13], i_struct_T *solution,
                   e_struct_T *memspace, k_struct_T *workingset, f_struct_T
                   *qrmanager, g_struct_T *cholmanager, struct_T *objective,
                   h_struct_T *options, int runTimeOptions_MaxIterations)
{
  int TYPE;
  int idx;
  int idxEndIneq;
  int idxStartIneq;
  int k;
  int nVar;
  bool guard1 = false;
  solution->iterations = 0;
  nVar = workingset->nVar - 1;
  guard1 = false;
  if (workingset->probType == 3) {
    idxEndIneq = workingset->sizes[0];
    for (idx = 0; idx < idxEndIneq; idx++) {
      solution->xstar[workingset->indexFixed[idx] - 1] = workingset->
        ub[workingset->indexFixed[idx] - 1];
    }

    idxEndIneq = workingset->sizes[3];
    for (idx = 0; idx < idxEndIneq; idx++) {
      if (workingset->isActiveConstr[(workingset->isActiveIdx[3] + idx) - 1]) {
        solution->xstar[workingset->indexLB[idx] - 1] = -workingset->
          lb[workingset->indexLB[idx] - 1];
      }
    }

    idxEndIneq = workingset->sizes[4];
    for (idx = 0; idx < idxEndIneq; idx++) {
      if (workingset->isActiveConstr[(workingset->isActiveIdx[4] + idx) - 1]) {
        solution->xstar[workingset->indexUB[idx] - 1] = workingset->
          ub[workingset->indexUB[idx] - 1];
      }
    }

    PresolveWorkingSet(solution, memspace, workingset, qrmanager);
    if (solution->state >= 0) {
      guard1 = true;
    }
  } else {
    solution->state = 82;
    guard1 = true;
  }

  if (guard1) {
    solution->iterations = 0;
    solution->maxConstr = b_maxConstraintViolation(workingset, solution->xstar);
    if (solution->maxConstr > 1.0E-6) {
      int PROBTYPE_ORIG;
      int b_nVar;
      PROBTYPE_ORIG = workingset->probType;
      b_nVar = workingset->nVar;
      solution->xstar[12] = solution->maxConstr + 1.0;
      if (workingset->probType == 3) {
        idxEndIneq = 1;
      } else {
        idxEndIneq = 4;
      }

      setProblemType(workingset, idxEndIneq);
      idxStartIneq = (workingset->nWConstr[0] + workingset->nWConstr[1]) + 1;
      idxEndIneq = workingset->nActiveConstr;
      for (TYPE = idxStartIneq; TYPE <= idxEndIneq; TYPE++) {
        workingset->isActiveConstr[(workingset->isActiveIdx[workingset->Wid[TYPE
          - 1] - 1] + workingset->Wlocalidx[TYPE - 1]) - 2] = false;
      }

      workingset->nWConstr[2] = 0;
      workingset->nWConstr[3] = 0;
      workingset->nWConstr[4] = 0;
      workingset->nActiveConstr = workingset->nWConstr[0] + workingset->
        nWConstr[1];
      objective->prev_objtype = objective->objtype;
      objective->prev_nvar = objective->nvar;
      objective->prev_hasLinear = objective->hasLinear;
      objective->objtype = 5;
      objective->nvar = 13;
      objective->gammaScalar = 1.0;
      objective->hasLinear = true;
      solution->fstar = computeFval(objective, memspace->workspace_double, H, f,
        solution->xstar);
      solution->state = 5;
      iterate(H, f, solution, memspace, workingset, qrmanager, cholmanager,
              objective, options->SolverName, 1.4901161193847657E-10, 1.0E-6,
              runTimeOptions_MaxIterations);
      if (workingset->isActiveConstr[(workingset->isActiveIdx[3] +
           workingset->sizes[3]) - 2]) {
        bool exitg1;
        idx = workingset->sizes[0];
        exitg1 = false;
        while ((!exitg1) && (idx + 1 <= workingset->nActiveConstr)) {
          if ((workingset->Wid[idx] == 4) && (workingset->Wlocalidx[idx] ==
               workingset->sizes[3])) {
            TYPE = workingset->Wid[idx] - 1;
            workingset->isActiveConstr[(workingset->isActiveIdx[workingset->
              Wid[idx] - 1] + workingset->Wlocalidx[idx]) - 2] = false;
            workingset->Wid[idx] = workingset->Wid[workingset->nActiveConstr - 1];
            workingset->Wlocalidx[idx] = workingset->Wlocalidx
              [workingset->nActiveConstr - 1];
            idxEndIneq = workingset->nVar;
            for (idxStartIneq = 0; idxStartIneq < idxEndIneq; idxStartIneq++) {
              workingset->ATwset[idxStartIneq + 13 * idx] = workingset->
                ATwset[idxStartIneq + 13 * (workingset->nActiveConstr - 1)];
            }

            workingset->bwset[idx] = workingset->bwset[workingset->nActiveConstr
              - 1];
            workingset->nActiveConstr--;
            workingset->nWConstr[TYPE]--;
            exitg1 = true;
          } else {
            idx++;
          }
        }
      }

      idxStartIneq = workingset->nActiveConstr - 1;
      while ((idxStartIneq + 1 > workingset->sizes[0]) && (idxStartIneq + 1 >
              b_nVar)) {
        TYPE = workingset->Wid[idxStartIneq] - 1;
        workingset->isActiveConstr[(workingset->isActiveIdx[workingset->
          Wid[idxStartIneq] - 1] + workingset->Wlocalidx[idxStartIneq]) - 2] =
          false;
        workingset->Wid[idxStartIneq] = workingset->Wid
          [workingset->nActiveConstr - 1];
        workingset->Wlocalidx[idxStartIneq] = workingset->Wlocalidx
          [workingset->nActiveConstr - 1];
        idxEndIneq = workingset->nVar;
        for (idx = 0; idx < idxEndIneq; idx++) {
          workingset->ATwset[idx + 13 * idxStartIneq] = workingset->ATwset[idx +
            13 * (workingset->nActiveConstr - 1)];
        }

        workingset->bwset[idxStartIneq] = workingset->bwset
          [workingset->nActiveConstr - 1];
        workingset->nActiveConstr--;
        workingset->nWConstr[TYPE]--;
        idxStartIneq--;
      }

      solution->maxConstr = solution->xstar[12];
      setProblemType(workingset, PROBTYPE_ORIG);
      objective->objtype = objective->prev_objtype;
      objective->nvar = objective->prev_nvar;
      objective->hasLinear = objective->prev_hasLinear;
      options->ObjectiveLimit = rtMinusInf;
      options->StepTolerance = 1.0E-6;
      if (solution->state != 0) {
        solution->maxConstr = b_maxConstraintViolation(workingset,
          solution->xstar);
        if (solution->maxConstr > 1.0E-6) {
          memset(&solution->lambda[0], 0, 25U * sizeof(double));
          solution->fstar = computeFval(objective, memspace->workspace_double, H,
            f, solution->xstar);
          solution->state = -2;
        } else {
          if (solution->maxConstr > 0.0) {
            double maxConstr_new;
            if (nVar + 1 < 400) {
              if (nVar >= 0) {
                memcpy(&solution->searchDir[0], &solution->xstar[0], (nVar + 1) *
                       sizeof(double));
              }
            } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

              for (k = 0; k <= nVar; k++) {
                solution->searchDir[k] = solution->xstar[k];
              }
            }

            PresolveWorkingSet(solution, memspace, workingset, qrmanager);
            maxConstr_new = b_maxConstraintViolation(workingset, solution->xstar);
            if (maxConstr_new >= solution->maxConstr) {
              solution->maxConstr = maxConstr_new;
              if (nVar + 1 < 400) {
                if (nVar >= 0) {
                  memcpy(&solution->xstar[0], &solution->searchDir[0], (nVar + 1)
                         * sizeof(double));
                }
              } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

                for (k = 0; k <= nVar; k++) {
                  solution->xstar[k] = solution->searchDir[k];
                }
              }
            }
          }

          iterate(H, f, solution, memspace, workingset, qrmanager, cholmanager,
                  objective, options->SolverName, options->StepTolerance,
                  options->ObjectiveLimit, runTimeOptions_MaxIterations);
        }
      }
    } else {
      iterate(H, f, solution, memspace, workingset, qrmanager, cholmanager,
              objective, options->SolverName, options->StepTolerance,
              options->ObjectiveLimit, runTimeOptions_MaxIterations);
    }
  }
}

/*
 * Arguments    : int n
 *                const double x[6]
 *                int ix0
 * Return Type  : double
 */
static double e_xnrm2(int n, const double x[6], int ix0)
{
  double scale;
  double y;
  int k;
  int kend;
  y = 0.0;
  scale = 3.3121686421112381E-170;
  kend = (ix0 + n) - 1;
  for (k = ix0; k <= kend; k++) {
    double absxk;
    absxk = fabs(x[k - 1]);
    if (absxk > scale) {
      double t;
      t = scale / absxk;
      y = y * t * t + 1.0;
      scale = absxk;
    } else {
      double t;
      t = absxk / scale;
      y += t * t;
    }
  }

  return scale * sqrt(y);
}

/*
 * Arguments    : const d_struct_T *obj_objfun_workspace
 *                const double x[12]
 *                double *fval
 *                int *status
 * Return Type  : void
 */
static void evalObjAndConstr(const d_struct_T *obj_objfun_workspace, const
  double x[12], double *fval, int *status)
{
  double a;
  double a_tmp;
  double ab_a;
  double ac_a;
  double b_a;
  double b_a_tmp;
  double b_x;
  double bb_a;
  double bc_a;
  double c_a;
  double c_a_tmp;
  double cb_a;
  double d_a;
  double d_a_tmp;
  double db_a;
  double e_a;
  double eb_a;
  double f_a;
  double fb_a;
  double g_a;
  double gb_a;
  double h_a;
  double hb_a;
  double i_a;
  double ib_a;
  double j_a;
  double jb_a;
  double k_a;
  double kb_a;
  double l_a;
  double lb_a;
  double m_a;
  double mb_a;
  double n_a;
  double nb_a;
  double o_a;
  double ob_a;
  double p_a;
  double pb_a;
  double q_a;
  double qb_a;
  double r_a;
  double rb_a;
  double s_a;
  double sb_a;
  double t_a;
  double tb_a;
  double u_a;
  double ub_a;
  double v_a;
  double vb_a;
  double w_a;
  double wb_a;
  double x_a;
  double xb_a;
  double y_a;
  double yb_a;

  /*      cost = gamma_quadratic*(W_act_motor^2*(u_in(1) - desired_motor_value/gain_motor)^2 + W_act_motor^2*(u_in(2) - desired_motor_value/gain_motor)^2 + W_act_motor^2*(u_in(3) - desired_motor_value/gain_motor)^2 + W_act_motor^2*(u_in(4) - desired_motor_value/gain_motor)^2 + W_act_tilt_el^2*(u_in(5) - desired_el_value/gain_el)^2 + W_act_tilt_el^2*(u_in(6) - desired_el_value/gain_el)^2 + W_act_tilt_el^2*(u_in(7) - desired_el_value/gain_el)^2 + W_act_tilt_el^2*(u_in(8) - desired_el_value/gain_el)^2 + W_act_tilt_az^2*(u_in(9) - desired_az_value/gain_az)^2 + W_act_tilt_az^2*(u_in(10) - desired_az_value/gain_az)^2 + W_act_tilt_az^2*(u_in(11) - desired_az_value/gain_az)^2 + W_act_tilt_az^2*(u_in(12) - desired_az_value/gain_az)^2) + W_dv_1^2*(dv_global(1) + (K_p_T*gain_motor^2*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta))*(cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)*u_in(4)^2) + K_p_T*gain_motor^2*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta))*(cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*u_in(4)^2) + K_p_T*gain_motor^2*cos(Psi)*cos(Theta)*(sin(u_in(5)*gain_el)*u_in(1)^2 + sin(u_in(6)*gain_el)*u_in(2)^2 + sin(u_in(7)*gain_el)*u_in(3)^2 + sin(u_in(8)*gain_el)*u_in(4)^2))/m)^2 + (W_dv_3^2*((100*(K_p_T*gain_motor^2*(cos(Psi)*sin(Phi) - cos(Phi)*sin(Psi)*sin(Theta))*(sin(u_in(5)*gain_el)*u_in(1)^2 + sin(u_in(6)*gain_el)*u_in(2)^2 + sin(u_in(7)*gain_el)*u_in(3)^2 + sin(u_in(8)*gain_el)*u_in(4)^2) + K_p_T*gain_motor^2*cos(Theta)*sin(Phi)*(cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*u_in(4)^2) - K_p_T*gain_motor^2*cos(Phi)*cos(Theta)*(cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)*u_in(4)^2)))/m - 100*dv_global(3) + 981)^2)/10000 + W_dv_2^2*(dv_global(2) - (K_p_T*gain_motor^2*(cos(Psi)*sin(Phi) - cos(Phi)*sin(Psi)*sin(Theta))*(cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)*u_in(4)^2) + K_p_T*gain_motor^2*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta))*(cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*u_in(4)^2) - K_p_T*gain_motor^2*cos(Theta)*sin(Psi)*(sin(u_in(5)*gain_el)*u_in(1)^2 + sin(u_in(6)*gain_el)*u_in(2)^2 + sin(u_in(7)*gain_el)*u_in(3)^2 + sin(u_in(8)*gain_el)*u_in(4)^2))/m)^2 + (W_dv_6^2*(I_zz*dv_global(6) - I_xx*p*q + I_yy*p*q + K_p_T*u_in(1)^2*gain_motor^2*l_1*sin(u_in(5)*gain_el) - K_p_T*u_in(2)^2*gain_motor^2*l_1*sin(u_in(6)*gain_el) - K_p_T*u_in(3)^2*gain_motor^2*l_2*sin(u_in(7)*gain_el) + K_p_T*u_in(4)^2*gain_motor^2*l_2*sin(u_in(8)*gain_el) - K_p_M*u_in(1)^2*gain_motor^2*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az) + K_p_M*u_in(2)^2*gain_motor^2*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az) - K_p_M*u_in(3)^2*gain_motor^2*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az) + K_p_M*u_in(4)^2*gain_motor^2*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az) - K_p_T*u_in(1)^2*gain_motor^2*l_4*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az) - K_p_T*u_in(2)^2*gain_motor^2*l_4*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az) + K_p_T*u_in(3)^2*gain_motor^2*l_3*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az) + K_p_T*u_in(4)^2*gain_motor^2*l_3*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az))^2)/I_zz^2 + (W_dv_4^2*(I_yy*q*r - I_xx*dv_global(4) - I_zz*q*r + K_p_M*u_in(1)^2*gain_motor^2*sin(u_in(5)*gain_el) - K_p_M*u_in(2)^2*gain_motor^2*sin(u_in(6)*gain_el) + K_p_M*u_in(3)^2*gain_motor^2*sin(u_in(7)*gain_el) - K_p_M*u_in(4)^2*gain_motor^2*sin(u_in(8)*gain_el) + K_p_T*u_in(1)^2*gain_motor^2*l_1*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az) - K_p_T*u_in(2)^2*gain_motor^2*l_1*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az) - K_p_T*u_in(3)^2*gain_motor^2*l_2*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az) + K_p_T*u_in(4)^2*gain_motor^2*l_2*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az) + K_p_T*u_in(1)^2*gain_motor^2*l_z*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az) + K_p_T*u_in(2)^2*gain_motor^2*l_z*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az) + K_p_T*u_in(3)^2*gain_motor^2*l_z*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az) + K_p_T*u_in(4)^2*gain_motor^2*l_z*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az))^2)/I_xx^2 + (W_dv_5^2*(2*I_zz*p*r - 2*I_xx*p*r - 2*I_yy*dv_global(5) + 2*K_p_T*u_in(1)^2*gain_motor^2*l_z*sin(u_in(5)*gain_el) + 2*K_p_T*u_in(2)^2*gain_motor^2*l_z*sin(u_in(6)*gain_el) + 2*K_p_T*u_in(3)^2*gain_motor^2*l_z*sin(u_in(7)*gain_el) + 2*K_p_T*u_in(4)^2*gain_motor^2*l_z*sin(u_in(8)*gain_el) - 2*K_p_M*u_in(1)^2*gain_motor^2*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az) + 2*K_p_M*u_in(2)^2*gain_motor^2*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az) - 2*K_p_M*u_in(3)^2*gain_motor^2*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az) + 2*K_p_M*u_in(4)^2*gain_motor^2*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az) + Cm_zero*S*V^2*rho*wing_chord + 2*K_p_T*u_in(1)^2*gain_motor^2*l_4*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az) + 2*K_p_T*u_in(2)^2*gain_motor^2*l_4*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az) - 2*K_p_T*u_in(3)^2*gain_motor^2*l_3*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az) - 2*K_p_T*u_in(4)^2*gain_motor^2*l_3*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az) + Cm_alpha*S*Theta*V^2*rho*wing_chord - Cm_alpha*S*V^2*flight_path_angle*rho*wing_chord)^2)/(4*I_yy^2);  */
  /*      %no aerodynamic on forces: */
  /*      cost = W_dv_1^2*gamma_quadratic_dv*(dv_global(1) + (K_p_T*gain_motor^2*(sin(Phi)*sin(Psi) + cos(Phi)*cos(Psi)*sin(Theta))*(cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)*u_in(4)^2) + K_p_T*gain_motor^2*(cos(Phi)*sin(Psi) - cos(Psi)*sin(Phi)*sin(Theta))*(cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*u_in(4)^2) + K_p_T*gain_motor^2*cos(Psi)*cos(Theta)*(sin(u_in(5)*gain_el)*u_in(1)^2 + sin(u_in(6)*gain_el)*u_in(2)^2 + sin(u_in(7)*gain_el)*u_in(3)^2 + sin(u_in(8)*gain_el)*u_in(4)^2))/m)^2 + (W_dv_3^2*gamma_quadratic_dv*((100*(K_p_T*gain_motor^2*(cos(Psi)*sin(Phi) - cos(Phi)*sin(Psi)*sin(Theta))*(sin(u_in(5)*gain_el)*u_in(1)^2 + sin(u_in(6)*gain_el)*u_in(2)^2 + sin(u_in(7)*gain_el)*u_in(3)^2 + sin(u_in(8)*gain_el)*u_in(4)^2) + K_p_T*gain_motor^2*cos(Theta)*sin(Phi)*(cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*u_in(4)^2) - K_p_T*gain_motor^2*cos(Phi)*cos(Theta)*(cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)*u_in(4)^2)))/m - 100*dv_global(3) + 981)^2)/10000 + W_dv_2^2*gamma_quadratic_dv*(dv_global(2) - (K_p_T*gain_motor^2*(cos(Psi)*sin(Phi) - cos(Phi)*sin(Psi)*sin(Theta))*(cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az)*u_in(4)^2) + K_p_T*gain_motor^2*(cos(Phi)*cos(Psi) + sin(Phi)*sin(Psi)*sin(Theta))*(cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az)*u_in(1)^2 + cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az)*u_in(2)^2 + cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az)*u_in(3)^2 + cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az)*u_in(4)^2) - K_p_T*gain_motor^2*cos(Theta)*sin(Psi)*(sin(u_in(5)*gain_el)*u_in(1)^2 + sin(u_in(6)*gain_el)*u_in(2)^2 + sin(u_in(7)*gain_el)*u_in(3)^2 + sin(u_in(8)*gain_el)*u_in(4)^2))/m)^2 + W_act_motor^2*gamma_quadratic_du*(u_in(1) - desired_motor_value/gain_motor)^2 + W_act_motor^2*gamma_quadratic_du*(u_in(2) - desired_motor_value/gain_motor)^2 + W_act_motor^2*gamma_quadratic_du*(u_in(3) - desired_motor_value/gain_motor)^2 + W_act_motor^2*gamma_quadratic_du*(u_in(4) - desired_motor_value/gain_motor)^2 + W_act_tilt_el^2*gamma_quadratic_du*(u_in(5) - desired_el_value/gain_el)^2 + W_act_tilt_el^2*gamma_quadratic_du*(u_in(6) - desired_el_value/gain_el)^2 + W_act_tilt_el^2*gamma_quadratic_du*(u_in(7) - desired_el_value/gain_el)^2 + W_act_tilt_el^2*gamma_quadratic_du*(u_in(8) - desired_el_value/gain_el)^2 + W_act_tilt_az^2*gamma_quadratic_du*(u_in(9) - desired_az_value/gain_az)^2 + W_act_tilt_az^2*gamma_quadratic_du*(u_in(10) - desired_az_value/gain_az)^2 + W_act_tilt_az^2*gamma_quadratic_du*(u_in(11) - desired_az_value/gain_az)^2 + W_act_tilt_az^2*gamma_quadratic_du*(u_in(12) - desired_az_value/gain_az)^2 + (W_dv_5^2*gamma_quadratic_dv*(2*I_zz*p*r - 2*I_xx*p*r - 2*I_yy*dv_global(5) + 2*K_p_T*u_in(1)^2*gain_motor^2*l_z*sin(u_in(5)*gain_el) + 2*K_p_T*u_in(2)^2*gain_motor^2*l_z*sin(u_in(6)*gain_el) + 2*K_p_T*u_in(3)^2*gain_motor^2*l_z*sin(u_in(7)*gain_el) + 2*K_p_T*u_in(4)^2*gain_motor^2*l_z*sin(u_in(8)*gain_el) - 2*K_p_M*u_in(1)^2*gain_motor^2*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az) + 2*K_p_M*u_in(2)^2*gain_motor^2*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az) - 2*K_p_M*u_in(3)^2*gain_motor^2*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az) + 2*K_p_M*u_in(4)^2*gain_motor^2*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az) + Cm_zero*S*V^2*rho*wing_chord + 2*K_p_T*u_in(1)^2*gain_motor^2*l_4*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az) + 2*K_p_T*u_in(2)^2*gain_motor^2*l_4*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az) - 2*K_p_T*u_in(3)^2*gain_motor^2*l_3*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az) - 2*K_p_T*u_in(4)^2*gain_motor^2*l_3*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az) + Cm_alpha*S*Theta*V^2*rho*wing_chord - Cm_alpha*S*V^2*flight_path_angle*rho*wing_chord)^2)/(4*I_yy^2) + (W_dv_6^2*gamma_quadratic_dv*(I_zz*dv_global(6) - I_xx*p*q + I_yy*p*q + K_p_T*u_in(1)^2*gain_motor^2*l_1*sin(u_in(5)*gain_el) - K_p_T*u_in(2)^2*gain_motor^2*l_1*sin(u_in(6)*gain_el) - K_p_T*u_in(3)^2*gain_motor^2*l_2*sin(u_in(7)*gain_el) + K_p_T*u_in(4)^2*gain_motor^2*l_2*sin(u_in(8)*gain_el) - K_p_M*u_in(1)^2*gain_motor^2*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az) + K_p_M*u_in(2)^2*gain_motor^2*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az) - K_p_M*u_in(3)^2*gain_motor^2*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az) + K_p_M*u_in(4)^2*gain_motor^2*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az) - K_p_T*u_in(1)^2*gain_motor^2*l_4*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az) - K_p_T*u_in(2)^2*gain_motor^2*l_4*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az) + K_p_T*u_in(3)^2*gain_motor^2*l_3*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az) + K_p_T*u_in(4)^2*gain_motor^2*l_3*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az))^2)/I_zz^2 + (W_dv_4^2*gamma_quadratic_dv*(I_yy*q*r - I_xx*dv_global(4) - I_zz*q*r + K_p_M*u_in(1)^2*gain_motor^2*sin(u_in(5)*gain_el) - K_p_M*u_in(2)^2*gain_motor^2*sin(u_in(6)*gain_el) + K_p_M*u_in(3)^2*gain_motor^2*sin(u_in(7)*gain_el) - K_p_M*u_in(4)^2*gain_motor^2*sin(u_in(8)*gain_el) + K_p_T*u_in(1)^2*gain_motor^2*l_1*cos(u_in(5)*gain_el)*cos(u_in(9)*gain_az) - K_p_T*u_in(2)^2*gain_motor^2*l_1*cos(u_in(6)*gain_el)*cos(u_in(10)*gain_az) - K_p_T*u_in(3)^2*gain_motor^2*l_2*cos(u_in(7)*gain_el)*cos(u_in(11)*gain_az) + K_p_T*u_in(4)^2*gain_motor^2*l_2*cos(u_in(8)*gain_el)*cos(u_in(12)*gain_az) + K_p_T*u_in(1)^2*gain_motor^2*l_z*cos(u_in(5)*gain_el)*sin(u_in(9)*gain_az) + K_p_T*u_in(2)^2*gain_motor^2*l_z*cos(u_in(6)*gain_el)*sin(u_in(10)*gain_az) + K_p_T*u_in(3)^2*gain_motor^2*l_z*cos(u_in(7)*gain_el)*sin(u_in(11)*gain_az) + K_p_T*u_in(4)^2*gain_motor^2*l_z*cos(u_in(8)*gain_el)*sin(u_in(12)*gain_az))^2)/I_xx^2; */
  a = obj_objfun_workspace->W_act_motor->contents;
  b_a = x[0] - obj_objfun_workspace->desired_motor_value->contents /
    obj_objfun_workspace->gain_motor->contents;
  c_a = obj_objfun_workspace->W_act_motor->contents;
  d_a = x[1] - obj_objfun_workspace->desired_motor_value->contents /
    obj_objfun_workspace->gain_motor->contents;
  e_a = obj_objfun_workspace->W_act_motor->contents;
  f_a = x[2] - obj_objfun_workspace->desired_motor_value->contents /
    obj_objfun_workspace->gain_motor->contents;
  g_a = obj_objfun_workspace->W_act_motor->contents;
  h_a = x[3] - obj_objfun_workspace->desired_motor_value->contents /
    obj_objfun_workspace->gain_motor->contents;
  i_a = obj_objfun_workspace->W_dv_2->contents;
  j_a = obj_objfun_workspace->V->contents;
  k_a = obj_objfun_workspace->Cl_alpha->contents;
  l_a = obj_objfun_workspace->Theta->contents;
  m_a = obj_objfun_workspace->Cl_alpha->contents;
  n_a = obj_objfun_workspace->Cl_alpha->contents;
  o_a = obj_objfun_workspace->flight_path_angle->contents;
  p_a = obj_objfun_workspace->V->contents;
  q_a = obj_objfun_workspace->gain_motor->contents;
  r_a = obj_objfun_workspace->gain_motor->contents;
  s_a = obj_objfun_workspace->V->contents;
  t_a = obj_objfun_workspace->Cl_alpha->contents;
  u_a = obj_objfun_workspace->Theta->contents;
  v_a = obj_objfun_workspace->Cl_alpha->contents;
  w_a = obj_objfun_workspace->Cl_alpha->contents;
  x_a = obj_objfun_workspace->flight_path_angle->contents;
  y_a = obj_objfun_workspace->V->contents;
  ab_a = obj_objfun_workspace->gain_motor->contents;
  bb_a = obj_objfun_workspace->V->contents;
  cb_a = obj_objfun_workspace->Cl_alpha->contents;
  db_a = obj_objfun_workspace->Theta->contents;
  eb_a = obj_objfun_workspace->Cl_alpha->contents;
  fb_a = obj_objfun_workspace->Cl_alpha->contents;
  gb_a = obj_objfun_workspace->flight_path_angle->contents;
  b_x = obj_objfun_workspace->V->contents;
  if (!rtIsNaN(b_x)) {
    if (b_x < 0.0) {
      b_x = -1.0;
    } else {
      b_x = (b_x > 0.0);
    }
  }

  a_tmp = x[0] * x[0];
  b_a_tmp = x[1] * x[1];
  c_a_tmp = x[2] * x[2];
  d_a_tmp = x[3] * x[3];
  j_a = obj_objfun_workspace->dv_global->contents[1] -
    ((((((obj_objfun_workspace->S->contents * (j_a * j_a) *
          obj_objfun_workspace->rho->contents * cos(obj_objfun_workspace->
           Beta->contents) * sin(obj_objfun_workspace->Theta->contents -
           obj_objfun_workspace->flight_path_angle->contents) *
          (((obj_objfun_workspace->K_Cd->contents * (k_a * k_a) * (l_a * l_a) -
             2.0 * obj_objfun_workspace->K_Cd->contents * (m_a * m_a) *
             obj_objfun_workspace->Theta->contents *
             obj_objfun_workspace->flight_path_angle->contents) +
            obj_objfun_workspace->K_Cd->contents * (n_a * n_a) * (o_a * o_a)) +
           obj_objfun_workspace->Cd_zero->contents) / 2.0 +
          obj_objfun_workspace->Cl_alpha->contents * obj_objfun_workspace->
          S->contents * (p_a * p_a) * obj_objfun_workspace->rho->contents * cos
          (obj_objfun_workspace->Theta->contents -
           obj_objfun_workspace->flight_path_angle->contents) *
          (obj_objfun_workspace->Theta->contents -
           obj_objfun_workspace->flight_path_angle->contents) / 2.0) * (cos
          (obj_objfun_workspace->Psi->contents) * sin(obj_objfun_workspace->
           Phi->contents) - cos(obj_objfun_workspace->Phi->contents) * sin
          (obj_objfun_workspace->Psi->contents) * sin
          (obj_objfun_workspace->Theta->contents)) + obj_objfun_workspace->
         K_p_T->contents * (q_a * q_a) * (cos(obj_objfun_workspace->
           Psi->contents) * sin(obj_objfun_workspace->Phi->contents) - cos
          (obj_objfun_workspace->Phi->contents) * sin(obj_objfun_workspace->
           Psi->contents) * sin(obj_objfun_workspace->Theta->contents)) * (((cos
            (x[4] * obj_objfun_workspace->gain_el->contents) * cos(x[8] *
             obj_objfun_workspace->gain_az->contents) * a_tmp + cos(x[5] *
             obj_objfun_workspace->gain_el->contents) * cos(x[9] *
             obj_objfun_workspace->gain_az->contents) * b_a_tmp) + cos(x[6] *
            obj_objfun_workspace->gain_el->contents) * cos(x[10] *
            obj_objfun_workspace->gain_az->contents) * c_a_tmp) + cos(x[7] *
           obj_objfun_workspace->gain_el->contents) * cos(x[11] *
           obj_objfun_workspace->gain_az->contents) * d_a_tmp)) +
        obj_objfun_workspace->K_p_T->contents * (r_a * r_a) * (cos
         (obj_objfun_workspace->Phi->contents) * cos(obj_objfun_workspace->
          Psi->contents) + sin(obj_objfun_workspace->Phi->contents) * sin
         (obj_objfun_workspace->Psi->contents) * sin(obj_objfun_workspace->
          Theta->contents)) * (((cos(x[4] * obj_objfun_workspace->
            gain_el->contents) * sin(x[8] * obj_objfun_workspace->
            gain_az->contents) * a_tmp + cos(x[5] *
            obj_objfun_workspace->gain_el->contents) * sin(x[9] *
            obj_objfun_workspace->gain_az->contents) * b_a_tmp) + cos(x[6] *
           obj_objfun_workspace->gain_el->contents) * sin(x[10] *
           obj_objfun_workspace->gain_az->contents) * c_a_tmp) + cos(x[7] *
          obj_objfun_workspace->gain_el->contents) * sin(x[11] *
          obj_objfun_workspace->gain_az->contents) * d_a_tmp)) - cos
       (obj_objfun_workspace->Theta->contents) * sin(obj_objfun_workspace->
        Psi->contents) * b_x * (obj_objfun_workspace->S->contents * (s_a * s_a) *
        obj_objfun_workspace->rho->contents * cos(obj_objfun_workspace->
         Beta->contents) * cos(obj_objfun_workspace->Theta->contents -
         obj_objfun_workspace->flight_path_angle->contents) *
        (((obj_objfun_workspace->K_Cd->contents * (t_a * t_a) * (u_a * u_a) -
           2.0 * obj_objfun_workspace->K_Cd->contents * (v_a * v_a) *
           obj_objfun_workspace->Theta->contents *
           obj_objfun_workspace->flight_path_angle->contents) +
          obj_objfun_workspace->K_Cd->contents * (w_a * w_a) * (x_a * x_a)) +
         obj_objfun_workspace->Cd_zero->contents) / 2.0 -
        obj_objfun_workspace->Cl_alpha->contents * obj_objfun_workspace->
        S->contents * (y_a * y_a) * obj_objfun_workspace->rho->contents * sin
        (obj_objfun_workspace->Theta->contents -
         obj_objfun_workspace->flight_path_angle->contents) *
        (obj_objfun_workspace->Theta->contents -
         obj_objfun_workspace->flight_path_angle->contents) / 2.0)) -
      obj_objfun_workspace->K_p_T->contents * (ab_a * ab_a) * cos
      (obj_objfun_workspace->Theta->contents) * sin(obj_objfun_workspace->
       Psi->contents) * (((sin(x[4] * obj_objfun_workspace->gain_el->contents) *
         a_tmp + sin(x[5] * obj_objfun_workspace->gain_el->contents) * b_a_tmp)
        + sin(x[6] * obj_objfun_workspace->gain_el->contents) * c_a_tmp) + sin
       (x[7] * obj_objfun_workspace->gain_el->contents) * d_a_tmp)) -
     obj_objfun_workspace->S->contents * (bb_a * bb_a) *
     obj_objfun_workspace->rho->contents * sin(obj_objfun_workspace->
      Beta->contents) * (cos(obj_objfun_workspace->Phi->contents) * cos
      (obj_objfun_workspace->Psi->contents) + sin(obj_objfun_workspace->
       Phi->contents) * sin(obj_objfun_workspace->Psi->contents) * sin
      (obj_objfun_workspace->Theta->contents)) * (((obj_objfun_workspace->
        K_Cd->contents * (cb_a * cb_a) * (db_a * db_a) - 2.0 *
        obj_objfun_workspace->K_Cd->contents * (eb_a * eb_a) *
        obj_objfun_workspace->Theta->contents *
        obj_objfun_workspace->flight_path_angle->contents) +
       obj_objfun_workspace->K_Cd->contents * (fb_a * fb_a) * (gb_a * gb_a)) +
      obj_objfun_workspace->Cd_zero->contents) / 2.0) / obj_objfun_workspace->
    m->contents;
  k_a = obj_objfun_workspace->W_act_tilt_el->contents;
  l_a = x[4] - obj_objfun_workspace->desired_el_value->contents /
    obj_objfun_workspace->gain_el->contents;
  m_a = obj_objfun_workspace->W_act_tilt_el->contents;
  n_a = x[5] - obj_objfun_workspace->desired_el_value->contents /
    obj_objfun_workspace->gain_el->contents;
  o_a = obj_objfun_workspace->W_act_tilt_el->contents;
  p_a = x[6] - obj_objfun_workspace->desired_el_value->contents /
    obj_objfun_workspace->gain_el->contents;
  q_a = obj_objfun_workspace->W_act_tilt_el->contents;
  r_a = x[7] - obj_objfun_workspace->desired_el_value->contents /
    obj_objfun_workspace->gain_el->contents;
  s_a = obj_objfun_workspace->W_act_tilt_az->contents;
  t_a = x[8] - obj_objfun_workspace->desired_az_value->contents /
    obj_objfun_workspace->gain_az->contents;
  u_a = obj_objfun_workspace->W_act_tilt_az->contents;
  v_a = x[9] - obj_objfun_workspace->desired_az_value->contents /
    obj_objfun_workspace->gain_az->contents;
  w_a = obj_objfun_workspace->W_act_tilt_az->contents;
  x_a = x[10] - obj_objfun_workspace->desired_az_value->contents /
    obj_objfun_workspace->gain_az->contents;
  y_a = obj_objfun_workspace->W_act_tilt_az->contents;
  ab_a = x[11] - obj_objfun_workspace->desired_az_value->contents /
    obj_objfun_workspace->gain_az->contents;
  bb_a = obj_objfun_workspace->W_dv_3->contents;
  cb_a = obj_objfun_workspace->V->contents;
  db_a = obj_objfun_workspace->Cl_alpha->contents;
  eb_a = obj_objfun_workspace->Theta->contents;
  fb_a = obj_objfun_workspace->Cl_alpha->contents;
  gb_a = obj_objfun_workspace->Cl_alpha->contents;
  hb_a = obj_objfun_workspace->flight_path_angle->contents;
  ib_a = obj_objfun_workspace->V->contents;
  jb_a = obj_objfun_workspace->V->contents;
  kb_a = obj_objfun_workspace->Cl_alpha->contents;
  lb_a = obj_objfun_workspace->Theta->contents;
  mb_a = obj_objfun_workspace->Cl_alpha->contents;
  nb_a = obj_objfun_workspace->Cl_alpha->contents;
  ob_a = obj_objfun_workspace->flight_path_angle->contents;
  pb_a = obj_objfun_workspace->V->contents;
  qb_a = obj_objfun_workspace->gain_motor->contents;
  rb_a = obj_objfun_workspace->gain_motor->contents;
  sb_a = obj_objfun_workspace->gain_motor->contents;
  tb_a = obj_objfun_workspace->V->contents;
  ub_a = obj_objfun_workspace->Cl_alpha->contents;
  vb_a = obj_objfun_workspace->Theta->contents;
  wb_a = obj_objfun_workspace->Cl_alpha->contents;
  xb_a = obj_objfun_workspace->Cl_alpha->contents;
  yb_a = obj_objfun_workspace->flight_path_angle->contents;
  b_x = obj_objfun_workspace->V->contents;
  if (!rtIsNaN(b_x)) {
    if (b_x < 0.0) {
      b_x = -1.0;
    } else {
      b_x = (b_x > 0.0);
    }
  }

  cb_a = (100.0 * obj_objfun_workspace->dv_global->contents[2] + 100.0 *
          (((((cos(obj_objfun_workspace->Phi->contents) * cos
               (obj_objfun_workspace->Theta->contents) *
               (obj_objfun_workspace->S->contents * (cb_a * cb_a) *
                obj_objfun_workspace->rho->contents * cos
                (obj_objfun_workspace->Beta->contents) * sin
                (obj_objfun_workspace->Theta->contents -
                 obj_objfun_workspace->flight_path_angle->contents) *
                (((obj_objfun_workspace->K_Cd->contents * (db_a * db_a) * (eb_a *
    eb_a) - 2.0 * obj_objfun_workspace->K_Cd->contents * (fb_a * fb_a) *
                   obj_objfun_workspace->Theta->contents *
                   obj_objfun_workspace->flight_path_angle->contents) +
                  obj_objfun_workspace->K_Cd->contents * (gb_a * gb_a) * (hb_a *
    hb_a)) + obj_objfun_workspace->Cd_zero->contents) / 2.0 +
                obj_objfun_workspace->Cl_alpha->contents *
                obj_objfun_workspace->S->contents * (ib_a * ib_a) *
                obj_objfun_workspace->rho->contents * cos
                (obj_objfun_workspace->Theta->contents -
                 obj_objfun_workspace->flight_path_angle->contents) *
                (obj_objfun_workspace->Theta->contents -
                 obj_objfun_workspace->flight_path_angle->contents) / 2.0) - b_x
               * (obj_objfun_workspace->S->contents * (jb_a * jb_a) *
                  obj_objfun_workspace->rho->contents * cos
                  (obj_objfun_workspace->Beta->contents) * cos
                  (obj_objfun_workspace->Theta->contents -
                   obj_objfun_workspace->flight_path_angle->contents) *
                  (((obj_objfun_workspace->K_Cd->contents * (kb_a * kb_a) *
                     (lb_a * lb_a) - 2.0 * obj_objfun_workspace->K_Cd->contents *
                     (mb_a * mb_a) * obj_objfun_workspace->Theta->contents *
                     obj_objfun_workspace->flight_path_angle->contents) +
                    obj_objfun_workspace->K_Cd->contents * (nb_a * nb_a) * (ob_a
    * ob_a)) + obj_objfun_workspace->Cd_zero->contents) / 2.0 -
                  obj_objfun_workspace->Cl_alpha->contents *
                  obj_objfun_workspace->S->contents * (pb_a * pb_a) *
                  obj_objfun_workspace->rho->contents * sin
                  (obj_objfun_workspace->Theta->contents -
                   obj_objfun_workspace->flight_path_angle->contents) *
                  (obj_objfun_workspace->Theta->contents -
                   obj_objfun_workspace->flight_path_angle->contents) / 2.0) *
               (cos(obj_objfun_workspace->Psi->contents) * sin
                (obj_objfun_workspace->Phi->contents) - cos
                (obj_objfun_workspace->Phi->contents) * sin
                (obj_objfun_workspace->Psi->contents) * sin
                (obj_objfun_workspace->Theta->contents))) -
              obj_objfun_workspace->K_p_T->contents * (qb_a * qb_a) * (cos
    (obj_objfun_workspace->Psi->contents) * sin(obj_objfun_workspace->
    Phi->contents) - cos(obj_objfun_workspace->Phi->contents) * sin
    (obj_objfun_workspace->Psi->contents) * sin(obj_objfun_workspace->
    Theta->contents)) * (((sin(x[4] * obj_objfun_workspace->gain_el->contents) *
    (x[0] * x[0]) + sin(x[5] * obj_objfun_workspace->gain_el->contents) * (x[1] *
    x[1])) + sin(x[6] * obj_objfun_workspace->gain_el->contents) * (x[2] * x[2]))
    + sin(x[7] * obj_objfun_workspace->gain_el->contents) * (x[3] * x[3]))) -
             obj_objfun_workspace->K_p_T->contents * (rb_a * rb_a) * cos
             (obj_objfun_workspace->Theta->contents) * sin
             (obj_objfun_workspace->Phi->contents) * (((cos(x[4] *
    obj_objfun_workspace->gain_el->contents) * sin(x[8] *
    obj_objfun_workspace->gain_az->contents) * (x[0] * x[0]) + cos(x[5] *
    obj_objfun_workspace->gain_el->contents) * sin(x[9] *
    obj_objfun_workspace->gain_az->contents) * (x[1] * x[1])) + cos(x[6] *
    obj_objfun_workspace->gain_el->contents) * sin(x[10] *
    obj_objfun_workspace->gain_az->contents) * (x[2] * x[2])) + cos(x[7] *
    obj_objfun_workspace->gain_el->contents) * sin(x[11] *
    obj_objfun_workspace->gain_az->contents) * (x[3] * x[3]))) +
            obj_objfun_workspace->K_p_T->contents * (sb_a * sb_a) * cos
            (obj_objfun_workspace->Phi->contents) * cos
            (obj_objfun_workspace->Theta->contents) * (((cos(x[4] *
    obj_objfun_workspace->gain_el->contents) * cos(x[8] *
    obj_objfun_workspace->gain_az->contents) * (x[0] * x[0]) + cos(x[5] *
    obj_objfun_workspace->gain_el->contents) * cos(x[9] *
    obj_objfun_workspace->gain_az->contents) * (x[1] * x[1])) + cos(x[6] *
    obj_objfun_workspace->gain_el->contents) * cos(x[10] *
    obj_objfun_workspace->gain_az->contents) * (x[2] * x[2])) + cos(x[7] *
              obj_objfun_workspace->gain_el->contents) * cos(x[11] *
              obj_objfun_workspace->gain_az->contents) * (x[3] * x[3]))) +
           obj_objfun_workspace->S->contents * (tb_a * tb_a) *
           obj_objfun_workspace->rho->contents * sin(obj_objfun_workspace->
            Beta->contents) * cos(obj_objfun_workspace->Theta->contents) * sin
           (obj_objfun_workspace->Phi->contents) * (((obj_objfun_workspace->
              K_Cd->contents * (ub_a * ub_a) * (vb_a * vb_a) - 2.0 *
              obj_objfun_workspace->K_Cd->contents * (wb_a * wb_a) *
              obj_objfun_workspace->Theta->contents *
              obj_objfun_workspace->flight_path_angle->contents) +
             obj_objfun_workspace->K_Cd->contents * (xb_a * xb_a) * (yb_a * yb_a))
            + obj_objfun_workspace->Cd_zero->contents) / 2.0) /
          obj_objfun_workspace->m->contents) - 981.0;
  db_a = obj_objfun_workspace->W_dv_1->contents;
  eb_a = obj_objfun_workspace->V->contents;
  fb_a = obj_objfun_workspace->Cl_alpha->contents;
  gb_a = obj_objfun_workspace->Theta->contents;
  hb_a = obj_objfun_workspace->Cl_alpha->contents;
  ib_a = obj_objfun_workspace->Cl_alpha->contents;
  jb_a = obj_objfun_workspace->flight_path_angle->contents;
  kb_a = obj_objfun_workspace->V->contents;
  lb_a = obj_objfun_workspace->gain_motor->contents;
  mb_a = obj_objfun_workspace->V->contents;
  nb_a = obj_objfun_workspace->Cl_alpha->contents;
  ob_a = obj_objfun_workspace->Theta->contents;
  pb_a = obj_objfun_workspace->Cl_alpha->contents;
  qb_a = obj_objfun_workspace->Cl_alpha->contents;
  rb_a = obj_objfun_workspace->flight_path_angle->contents;
  sb_a = obj_objfun_workspace->V->contents;
  tb_a = obj_objfun_workspace->gain_motor->contents;
  ub_a = obj_objfun_workspace->gain_motor->contents;
  vb_a = obj_objfun_workspace->V->contents;
  wb_a = obj_objfun_workspace->Cl_alpha->contents;
  xb_a = obj_objfun_workspace->Theta->contents;
  yb_a = obj_objfun_workspace->Cl_alpha->contents;
  ac_a = obj_objfun_workspace->Cl_alpha->contents;
  bc_a = obj_objfun_workspace->flight_path_angle->contents;
  b_x = obj_objfun_workspace->V->contents;
  if (!rtIsNaN(b_x)) {
    if (b_x < 0.0) {
      b_x = -1.0;
    } else {
      b_x = (b_x > 0.0);
    }
  }

  eb_a = obj_objfun_workspace->dv_global->contents[0] +
    ((((((obj_objfun_workspace->S->contents * (eb_a * eb_a) *
          obj_objfun_workspace->rho->contents * cos(obj_objfun_workspace->
           Beta->contents) * sin(obj_objfun_workspace->Theta->contents -
           obj_objfun_workspace->flight_path_angle->contents) *
          (((obj_objfun_workspace->K_Cd->contents * (fb_a * fb_a) * (gb_a * gb_a)
             - 2.0 * obj_objfun_workspace->K_Cd->contents * (hb_a * hb_a) *
             obj_objfun_workspace->Theta->contents *
             obj_objfun_workspace->flight_path_angle->contents) +
            obj_objfun_workspace->K_Cd->contents * (ib_a * ib_a) * (jb_a * jb_a))
           + obj_objfun_workspace->Cd_zero->contents) / 2.0 +
          obj_objfun_workspace->Cl_alpha->contents * obj_objfun_workspace->
          S->contents * (kb_a * kb_a) * obj_objfun_workspace->rho->contents *
          cos(obj_objfun_workspace->Theta->contents -
              obj_objfun_workspace->flight_path_angle->contents) *
          (obj_objfun_workspace->Theta->contents -
           obj_objfun_workspace->flight_path_angle->contents) / 2.0) * (sin
          (obj_objfun_workspace->Phi->contents) * sin(obj_objfun_workspace->
           Psi->contents) + cos(obj_objfun_workspace->Phi->contents) * cos
          (obj_objfun_workspace->Psi->contents) * sin
          (obj_objfun_workspace->Theta->contents)) + obj_objfun_workspace->
         K_p_T->contents * (lb_a * lb_a) * (sin(obj_objfun_workspace->
           Phi->contents) * sin(obj_objfun_workspace->Psi->contents) + cos
          (obj_objfun_workspace->Phi->contents) * cos(obj_objfun_workspace->
           Psi->contents) * sin(obj_objfun_workspace->Theta->contents)) * (((cos
            (x[4] * obj_objfun_workspace->gain_el->contents) * cos(x[8] *
             obj_objfun_workspace->gain_az->contents) * (x[0] * x[0]) + cos(x[5]
             * obj_objfun_workspace->gain_el->contents) * cos(x[9] *
             obj_objfun_workspace->gain_az->contents) * (x[1] * x[1])) + cos(x[6]
            * obj_objfun_workspace->gain_el->contents) * cos(x[10] *
            obj_objfun_workspace->gain_az->contents) * (x[2] * x[2])) + cos(x[7]
           * obj_objfun_workspace->gain_el->contents) * cos(x[11] *
           obj_objfun_workspace->gain_az->contents) * (x[3] * x[3]))) + cos
        (obj_objfun_workspace->Psi->contents) * cos(obj_objfun_workspace->
         Theta->contents) * b_x * (obj_objfun_workspace->S->contents * (mb_a *
          mb_a) * obj_objfun_workspace->rho->contents * cos
         (obj_objfun_workspace->Beta->contents) * cos
         (obj_objfun_workspace->Theta->contents -
          obj_objfun_workspace->flight_path_angle->contents) *
         (((obj_objfun_workspace->K_Cd->contents * (nb_a * nb_a) * (ob_a * ob_a)
            - 2.0 * obj_objfun_workspace->K_Cd->contents * (pb_a * pb_a) *
            obj_objfun_workspace->Theta->contents *
            obj_objfun_workspace->flight_path_angle->contents) +
           obj_objfun_workspace->K_Cd->contents * (qb_a * qb_a) * (rb_a * rb_a))
          + obj_objfun_workspace->Cd_zero->contents) / 2.0 -
         obj_objfun_workspace->Cl_alpha->contents * obj_objfun_workspace->
         S->contents * (sb_a * sb_a) * obj_objfun_workspace->rho->contents * sin
         (obj_objfun_workspace->Theta->contents -
          obj_objfun_workspace->flight_path_angle->contents) *
         (obj_objfun_workspace->Theta->contents -
          obj_objfun_workspace->flight_path_angle->contents) / 2.0)) +
       obj_objfun_workspace->K_p_T->contents * (tb_a * tb_a) * (cos
        (obj_objfun_workspace->Phi->contents) * sin(obj_objfun_workspace->
         Psi->contents) - cos(obj_objfun_workspace->Psi->contents) * sin
        (obj_objfun_workspace->Phi->contents) * sin(obj_objfun_workspace->
         Theta->contents)) * (((cos(x[4] * obj_objfun_workspace->
           gain_el->contents) * sin(x[8] * obj_objfun_workspace->
           gain_az->contents) * (x[0] * x[0]) + cos(x[5] *
           obj_objfun_workspace->gain_el->contents) * sin(x[9] *
           obj_objfun_workspace->gain_az->contents) * (x[1] * x[1])) + cos(x[6] *
          obj_objfun_workspace->gain_el->contents) * sin(x[10] *
          obj_objfun_workspace->gain_az->contents) * (x[2] * x[2])) + cos(x[7] *
         obj_objfun_workspace->gain_el->contents) * sin(x[11] *
         obj_objfun_workspace->gain_az->contents) * (x[3] * x[3]))) +
      obj_objfun_workspace->K_p_T->contents * (ub_a * ub_a) * cos
      (obj_objfun_workspace->Psi->contents) * cos(obj_objfun_workspace->
       Theta->contents) * (((sin(x[4] * obj_objfun_workspace->gain_el->contents)
         * (x[0] * x[0]) + sin(x[5] * obj_objfun_workspace->gain_el->contents) *
         (x[1] * x[1])) + sin(x[6] * obj_objfun_workspace->gain_el->contents) *
        (x[2] * x[2])) + sin(x[7] * obj_objfun_workspace->gain_el->contents) *
       (x[3] * x[3]))) - obj_objfun_workspace->S->contents * (vb_a * vb_a) *
     obj_objfun_workspace->rho->contents * sin(obj_objfun_workspace->
      Beta->contents) * (cos(obj_objfun_workspace->Phi->contents) * sin
      (obj_objfun_workspace->Psi->contents) - cos(obj_objfun_workspace->
       Psi->contents) * sin(obj_objfun_workspace->Phi->contents) * sin
      (obj_objfun_workspace->Theta->contents)) * (((obj_objfun_workspace->
        K_Cd->contents * (wb_a * wb_a) * (xb_a * xb_a) - 2.0 *
        obj_objfun_workspace->K_Cd->contents * (yb_a * yb_a) *
        obj_objfun_workspace->Theta->contents *
        obj_objfun_workspace->flight_path_angle->contents) +
       obj_objfun_workspace->K_Cd->contents * (ac_a * ac_a) * (bc_a * bc_a)) +
      obj_objfun_workspace->Cd_zero->contents) / 2.0) / obj_objfun_workspace->
    m->contents;
  fb_a = obj_objfun_workspace->W_dv_5->contents;
  gb_a = obj_objfun_workspace->gain_motor->contents;
  hb_a = obj_objfun_workspace->gain_motor->contents;
  ib_a = obj_objfun_workspace->gain_motor->contents;
  jb_a = obj_objfun_workspace->gain_motor->contents;
  kb_a = obj_objfun_workspace->gain_motor->contents;
  lb_a = obj_objfun_workspace->gain_motor->contents;
  mb_a = obj_objfun_workspace->gain_motor->contents;
  nb_a = obj_objfun_workspace->gain_motor->contents;
  ob_a = obj_objfun_workspace->V->contents;
  pb_a = obj_objfun_workspace->gain_motor->contents;
  qb_a = obj_objfun_workspace->gain_motor->contents;
  rb_a = obj_objfun_workspace->gain_motor->contents;
  sb_a = obj_objfun_workspace->gain_motor->contents;
  tb_a = obj_objfun_workspace->V->contents;
  ub_a = obj_objfun_workspace->V->contents;
  gb_a = ((((((((((((((((2.0 * obj_objfun_workspace->I_zz->contents *
    obj_objfun_workspace->p->contents * obj_objfun_workspace->r->contents - 2.0 *
    obj_objfun_workspace->I_xx->contents * obj_objfun_workspace->p->contents *
    obj_objfun_workspace->r->contents) - 2.0 * obj_objfun_workspace->
                        I_yy->contents * obj_objfun_workspace->
                        dv_global->contents[4]) + 2.0 *
                       obj_objfun_workspace->K_p_T->contents * a_tmp * (gb_a *
    gb_a) * obj_objfun_workspace->l_z->contents * sin(x[4] *
    obj_objfun_workspace->gain_el->contents)) + 2.0 *
                      obj_objfun_workspace->K_p_T->contents * b_a_tmp * (hb_a *
    hb_a) * obj_objfun_workspace->l_z->contents * sin(x[5] *
    obj_objfun_workspace->gain_el->contents)) + 2.0 *
                     obj_objfun_workspace->K_p_T->contents * c_a_tmp * (ib_a *
    ib_a) * obj_objfun_workspace->l_z->contents * sin(x[6] *
    obj_objfun_workspace->gain_el->contents)) + 2.0 *
                    obj_objfun_workspace->K_p_T->contents * d_a_tmp * (jb_a *
    jb_a) * obj_objfun_workspace->l_z->contents * sin(x[7] *
    obj_objfun_workspace->gain_el->contents)) - 2.0 *
                   obj_objfun_workspace->K_p_M->contents * a_tmp * (kb_a * kb_a)
                   * cos(x[4] * obj_objfun_workspace->gain_el->contents) * sin
                   (x[8] * obj_objfun_workspace->gain_az->contents)) + 2.0 *
                  obj_objfun_workspace->K_p_M->contents * b_a_tmp * (lb_a * lb_a)
                  * cos(x[5] * obj_objfun_workspace->gain_el->contents) * sin(x
    [9] * obj_objfun_workspace->gain_az->contents)) - 2.0 *
                 obj_objfun_workspace->K_p_M->contents * c_a_tmp * (mb_a * mb_a)
                 * cos(x[6] * obj_objfun_workspace->gain_el->contents) * sin(x
    [10] * obj_objfun_workspace->gain_az->contents)) + 2.0 *
                obj_objfun_workspace->K_p_M->contents * d_a_tmp * (nb_a * nb_a) *
                cos(x[7] * obj_objfun_workspace->gain_el->contents) * sin(x[11] *
    obj_objfun_workspace->gain_az->contents)) + obj_objfun_workspace->
               Cm_zero->contents * obj_objfun_workspace->S->contents * (ob_a *
    ob_a) * obj_objfun_workspace->rho->contents *
               obj_objfun_workspace->wing_chord->contents) + 2.0 *
              obj_objfun_workspace->K_p_T->contents * (x[0] * x[0]) * (pb_a *
    pb_a) * obj_objfun_workspace->l_4->contents * cos(x[4] *
    obj_objfun_workspace->gain_el->contents) * cos(x[8] *
    obj_objfun_workspace->gain_az->contents)) + 2.0 *
             obj_objfun_workspace->K_p_T->contents * (x[1] * x[1]) * (qb_a *
              qb_a) * obj_objfun_workspace->l_4->contents * cos(x[5] *
              obj_objfun_workspace->gain_el->contents) * cos(x[9] *
              obj_objfun_workspace->gain_az->contents)) - 2.0 *
            obj_objfun_workspace->K_p_T->contents * (x[2] * x[2]) * (rb_a * rb_a)
            * obj_objfun_workspace->l_3->contents * cos(x[6] *
             obj_objfun_workspace->gain_el->contents) * cos(x[10] *
             obj_objfun_workspace->gain_az->contents)) - 2.0 *
           obj_objfun_workspace->K_p_T->contents * (x[3] * x[3]) * (sb_a * sb_a)
           * obj_objfun_workspace->l_3->contents * cos(x[7] *
            obj_objfun_workspace->gain_el->contents) * cos(x[11] *
            obj_objfun_workspace->gain_az->contents)) +
          obj_objfun_workspace->Cm_alpha->contents * obj_objfun_workspace->
          S->contents * obj_objfun_workspace->Theta->contents * (tb_a * tb_a) *
          obj_objfun_workspace->rho->contents * obj_objfun_workspace->
          wing_chord->contents) - obj_objfun_workspace->Cm_alpha->contents *
    obj_objfun_workspace->S->contents * (ub_a * ub_a) *
    obj_objfun_workspace->flight_path_angle->contents *
    obj_objfun_workspace->rho->contents * obj_objfun_workspace->
    wing_chord->contents;
  hb_a = obj_objfun_workspace->I_yy->contents;
  ib_a = obj_objfun_workspace->W_dv_6->contents;
  jb_a = obj_objfun_workspace->gain_motor->contents;
  kb_a = obj_objfun_workspace->gain_motor->contents;
  lb_a = obj_objfun_workspace->gain_motor->contents;
  mb_a = obj_objfun_workspace->gain_motor->contents;
  nb_a = obj_objfun_workspace->gain_motor->contents;
  ob_a = obj_objfun_workspace->gain_motor->contents;
  pb_a = obj_objfun_workspace->gain_motor->contents;
  qb_a = obj_objfun_workspace->gain_motor->contents;
  rb_a = obj_objfun_workspace->gain_motor->contents;
  sb_a = obj_objfun_workspace->gain_motor->contents;
  tb_a = obj_objfun_workspace->gain_motor->contents;
  ub_a = obj_objfun_workspace->gain_motor->contents;
  jb_a = (((((((((((((obj_objfun_workspace->I_zz->contents *
                      obj_objfun_workspace->dv_global->contents[5] -
                      obj_objfun_workspace->I_xx->contents *
                      obj_objfun_workspace->p->contents *
                      obj_objfun_workspace->q->contents) +
                     obj_objfun_workspace->I_yy->contents *
                     obj_objfun_workspace->p->contents * obj_objfun_workspace->
                     q->contents) + obj_objfun_workspace->K_p_T->contents *
                    a_tmp * (jb_a * jb_a) * obj_objfun_workspace->l_1->contents *
                    sin(x[4] * obj_objfun_workspace->gain_el->contents)) -
                   obj_objfun_workspace->K_p_T->contents * b_a_tmp * (kb_a *
    kb_a) * obj_objfun_workspace->l_1->contents * sin(x[5] *
    obj_objfun_workspace->gain_el->contents)) - obj_objfun_workspace->
                  K_p_T->contents * c_a_tmp * (lb_a * lb_a) *
                  obj_objfun_workspace->l_2->contents * sin(x[6] *
    obj_objfun_workspace->gain_el->contents)) + obj_objfun_workspace->
                 K_p_T->contents * d_a_tmp * (mb_a * mb_a) *
                 obj_objfun_workspace->l_2->contents * sin(x[7] *
    obj_objfun_workspace->gain_el->contents)) - obj_objfun_workspace->
                K_p_M->contents * a_tmp * (nb_a * nb_a) * cos(x[4] *
    obj_objfun_workspace->gain_el->contents) * cos(x[8] *
    obj_objfun_workspace->gain_az->contents)) + obj_objfun_workspace->
               K_p_M->contents * b_a_tmp * (ob_a * ob_a) * cos(x[5] *
    obj_objfun_workspace->gain_el->contents) * cos(x[9] *
    obj_objfun_workspace->gain_az->contents)) - obj_objfun_workspace->
              K_p_M->contents * c_a_tmp * (pb_a * pb_a) * cos(x[6] *
    obj_objfun_workspace->gain_el->contents) * cos(x[10] *
    obj_objfun_workspace->gain_az->contents)) + obj_objfun_workspace->
             K_p_M->contents * d_a_tmp * (qb_a * qb_a) * cos(x[7] *
              obj_objfun_workspace->gain_el->contents) * cos(x[11] *
              obj_objfun_workspace->gain_az->contents)) -
            obj_objfun_workspace->K_p_T->contents * (x[0] * x[0]) * (rb_a * rb_a)
            * obj_objfun_workspace->l_4->contents * cos(x[4] *
             obj_objfun_workspace->gain_el->contents) * sin(x[8] *
             obj_objfun_workspace->gain_az->contents)) -
           obj_objfun_workspace->K_p_T->contents * (x[1] * x[1]) * (sb_a * sb_a)
           * obj_objfun_workspace->l_4->contents * cos(x[5] *
            obj_objfun_workspace->gain_el->contents) * sin(x[9] *
            obj_objfun_workspace->gain_az->contents)) +
          obj_objfun_workspace->K_p_T->contents * (x[2] * x[2]) * (tb_a * tb_a) *
          obj_objfun_workspace->l_3->contents * cos(x[6] *
           obj_objfun_workspace->gain_el->contents) * sin(x[10] *
           obj_objfun_workspace->gain_az->contents)) +
    obj_objfun_workspace->K_p_T->contents * (x[3] * x[3]) * (ub_a * ub_a) *
    obj_objfun_workspace->l_3->contents * cos(x[7] *
    obj_objfun_workspace->gain_el->contents) * sin(x[11] *
    obj_objfun_workspace->gain_az->contents);
  kb_a = obj_objfun_workspace->I_zz->contents;
  lb_a = obj_objfun_workspace->W_dv_4->contents;
  mb_a = obj_objfun_workspace->gain_motor->contents;
  nb_a = obj_objfun_workspace->gain_motor->contents;
  ob_a = obj_objfun_workspace->gain_motor->contents;
  pb_a = obj_objfun_workspace->gain_motor->contents;
  qb_a = obj_objfun_workspace->gain_motor->contents;
  rb_a = obj_objfun_workspace->gain_motor->contents;
  sb_a = obj_objfun_workspace->gain_motor->contents;
  tb_a = obj_objfun_workspace->gain_motor->contents;
  ub_a = obj_objfun_workspace->gain_motor->contents;
  vb_a = obj_objfun_workspace->gain_motor->contents;
  wb_a = obj_objfun_workspace->gain_motor->contents;
  xb_a = obj_objfun_workspace->gain_motor->contents;
  mb_a = (((((((((((((obj_objfun_workspace->I_yy->contents *
                      obj_objfun_workspace->q->contents *
                      obj_objfun_workspace->r->contents -
                      obj_objfun_workspace->I_xx->contents *
                      obj_objfun_workspace->dv_global->contents[3]) -
                     obj_objfun_workspace->I_zz->contents *
                     obj_objfun_workspace->q->contents * obj_objfun_workspace->
                     r->contents) + obj_objfun_workspace->K_p_M->contents * (x[0]
    * x[0]) * (mb_a * mb_a) * sin(x[4] * obj_objfun_workspace->gain_el->contents))
                   - obj_objfun_workspace->K_p_M->contents * (x[1] * x[1]) *
                   (nb_a * nb_a) * sin(x[5] * obj_objfun_workspace->
    gain_el->contents)) + obj_objfun_workspace->K_p_M->contents * (x[2] * x[2]) *
                  (ob_a * ob_a) * sin(x[6] * obj_objfun_workspace->
    gain_el->contents)) - obj_objfun_workspace->K_p_M->contents * (x[3] * x[3]) *
                 (pb_a * pb_a) * sin(x[7] * obj_objfun_workspace->
    gain_el->contents)) + obj_objfun_workspace->K_p_T->contents * (x[0] * x[0]) *
                (qb_a * qb_a) * obj_objfun_workspace->l_1->contents * cos(x[4] *
    obj_objfun_workspace->gain_el->contents) * cos(x[8] *
    obj_objfun_workspace->gain_az->contents)) - obj_objfun_workspace->
               K_p_T->contents * (x[1] * x[1]) * (rb_a * rb_a) *
               obj_objfun_workspace->l_1->contents * cos(x[5] *
    obj_objfun_workspace->gain_el->contents) * cos(x[9] *
    obj_objfun_workspace->gain_az->contents)) - obj_objfun_workspace->
              K_p_T->contents * (x[2] * x[2]) * (sb_a * sb_a) *
              obj_objfun_workspace->l_2->contents * cos(x[6] *
    obj_objfun_workspace->gain_el->contents) * cos(x[10] *
    obj_objfun_workspace->gain_az->contents)) + obj_objfun_workspace->
             K_p_T->contents * (x[3] * x[3]) * (tb_a * tb_a) *
             obj_objfun_workspace->l_2->contents * cos(x[7] *
              obj_objfun_workspace->gain_el->contents) * cos(x[11] *
              obj_objfun_workspace->gain_az->contents)) +
            obj_objfun_workspace->K_p_T->contents * (x[0] * x[0]) * (ub_a * ub_a)
            * obj_objfun_workspace->l_z->contents * cos(x[4] *
             obj_objfun_workspace->gain_el->contents) * sin(x[8] *
             obj_objfun_workspace->gain_az->contents)) +
           obj_objfun_workspace->K_p_T->contents * (x[1] * x[1]) * (vb_a * vb_a)
           * obj_objfun_workspace->l_z->contents * cos(x[5] *
            obj_objfun_workspace->gain_el->contents) * sin(x[9] *
            obj_objfun_workspace->gain_az->contents)) +
          obj_objfun_workspace->K_p_T->contents * (x[2] * x[2]) * (wb_a * wb_a) *
          obj_objfun_workspace->l_z->contents * cos(x[6] *
           obj_objfun_workspace->gain_el->contents) * sin(x[10] *
           obj_objfun_workspace->gain_az->contents)) +
    obj_objfun_workspace->K_p_T->contents * (x[3] * x[3]) * (xb_a * xb_a) *
    obj_objfun_workspace->l_z->contents * cos(x[7] *
    obj_objfun_workspace->gain_el->contents) * sin(x[11] *
    obj_objfun_workspace->gain_az->contents);
  nb_a = obj_objfun_workspace->I_xx->contents;
  *fval = ((((((((((((((((a * a * obj_objfun_workspace->
    gamma_quadratic_du->contents * (b_a * b_a) + c_a * c_a *
    obj_objfun_workspace->gamma_quadratic_du->contents * (d_a * d_a)) + e_a *
    e_a * obj_objfun_workspace->gamma_quadratic_du->contents * (f_a * f_a)) +
                        g_a * g_a * obj_objfun_workspace->
                        gamma_quadratic_du->contents * (h_a * h_a)) + i_a * i_a *
                       obj_objfun_workspace->gamma_quadratic_dv->contents * (j_a
    * j_a)) + k_a * k_a * obj_objfun_workspace->gamma_quadratic_du->contents *
                      (l_a * l_a)) + m_a * m_a *
                     obj_objfun_workspace->gamma_quadratic_du->contents * (n_a *
    n_a)) + o_a * o_a * obj_objfun_workspace->gamma_quadratic_du->contents *
                    (p_a * p_a)) + q_a * q_a *
                   obj_objfun_workspace->gamma_quadratic_du->contents * (r_a *
    r_a)) + s_a * s_a * obj_objfun_workspace->gamma_quadratic_du->contents *
                  (t_a * t_a)) + u_a * u_a *
                 obj_objfun_workspace->gamma_quadratic_du->contents * (v_a * v_a))
                + w_a * w_a * obj_objfun_workspace->gamma_quadratic_du->contents
                * (x_a * x_a)) + y_a * y_a *
               obj_objfun_workspace->gamma_quadratic_du->contents * (ab_a * ab_a))
              + bb_a * bb_a * obj_objfun_workspace->gamma_quadratic_dv->contents
              * (cb_a * cb_a) / 10000.0) + db_a * db_a *
             obj_objfun_workspace->gamma_quadratic_dv->contents * (eb_a * eb_a))
            + fb_a * fb_a * obj_objfun_workspace->gamma_quadratic_dv->contents *
            (gb_a * gb_a) / (4.0 * (hb_a * hb_a))) + ib_a * ib_a *
           obj_objfun_workspace->gamma_quadratic_dv->contents * (jb_a * jb_a) /
           (kb_a * kb_a)) + lb_a * lb_a *
    obj_objfun_workspace->gamma_quadratic_dv->contents * (mb_a * mb_a) / (nb_a *
    nb_a);
  *status = 1;
  if (rtIsInf(*fval) || rtIsNaN(*fval)) {
    if (rtIsNaN(*fval)) {
      *status = -3;
    } else if (*fval < 0.0) {
      *status = -1;
    } else {
      *status = -2;
    }
  }

  if (*status == 1) {
    *status = 1;
  }
}

/*
 * Arguments    : const d_struct_T *obj_objfun_workspace
 *                const double x[12]
 *                double grad_workspace[13]
 *                double *fval
 *                int *status
 * Return Type  : void
 */
static void evalObjAndConstrAndDerivatives(const d_struct_T
  *obj_objfun_workspace, const double x[12], double grad_workspace[13], double
  *fval, int *status)
{
  double varargout_2[12];
  c_compute_cost_and_gradient_fcn(obj_objfun_workspace->W_act_motor,
    obj_objfun_workspace->gamma_quadratic_du,
    obj_objfun_workspace->desired_motor_value, obj_objfun_workspace->gain_motor,
    obj_objfun_workspace->W_dv_2, obj_objfun_workspace->gamma_quadratic_dv,
    obj_objfun_workspace->dv_global, obj_objfun_workspace->S,
    obj_objfun_workspace->V, obj_objfun_workspace->rho,
    obj_objfun_workspace->Beta, obj_objfun_workspace->Theta,
    obj_objfun_workspace->flight_path_angle, obj_objfun_workspace->K_Cd,
    obj_objfun_workspace->Cl_alpha, obj_objfun_workspace->Cd_zero,
    obj_objfun_workspace->Psi, obj_objfun_workspace->Phi,
    obj_objfun_workspace->K_p_T, obj_objfun_workspace->gain_el,
    obj_objfun_workspace->gain_az, obj_objfun_workspace->m,
    obj_objfun_workspace->W_act_tilt_el, obj_objfun_workspace->desired_el_value,
    obj_objfun_workspace->W_act_tilt_az, obj_objfun_workspace->desired_az_value,
    obj_objfun_workspace->W_dv_3, obj_objfun_workspace->W_dv_1,
    obj_objfun_workspace->W_dv_5, obj_objfun_workspace->I_zz,
    obj_objfun_workspace->p, obj_objfun_workspace->r, obj_objfun_workspace->I_xx,
    obj_objfun_workspace->I_yy, obj_objfun_workspace->l_z,
    obj_objfun_workspace->K_p_M, obj_objfun_workspace->Cm_zero,
    obj_objfun_workspace->wing_chord, obj_objfun_workspace->l_4,
    obj_objfun_workspace->l_3, obj_objfun_workspace->Cm_alpha,
    obj_objfun_workspace->W_dv_6, obj_objfun_workspace->q,
    obj_objfun_workspace->l_1, obj_objfun_workspace->l_2,
    obj_objfun_workspace->W_dv_4, x, fval, varargout_2);
  memcpy(&grad_workspace[0], &varargout_2[0], 12U * sizeof(double));
  *status = 1;
  if (rtIsInf(*fval) || rtIsNaN(*fval)) {
    if (rtIsNaN(*fval)) {
      *status = -3;
    } else if (*fval < 0.0) {
      *status = -1;
    } else {
      *status = -2;
    }
  } else {
    int idx_current;
    bool allFinite;
    allFinite = true;
    idx_current = 0;
    while (allFinite && (idx_current + 1 <= 12)) {
      allFinite = ((!rtIsInf(grad_workspace[idx_current])) && (!rtIsNaN
        (grad_workspace[idx_current])));
      idx_current++;
    }

    if (!allFinite) {
      idx_current--;
      if (rtIsNaN(grad_workspace[idx_current])) {
        *status = -3;
      } else if (grad_workspace[idx_current] < 0.0) {
        *status = -1;
      } else {
        *status = -2;
      }
    }
  }

  if (*status == 1) {
    *status = 1;
  }
}

/*
 * Arguments    : f_struct_T *obj
 *                const double A[325]
 *                int mrows
 *                int ncols
 * Return Type  : void
 */
static void factorQR(f_struct_T *obj, const double A[325], int mrows, int ncols)
{
  int b_idx;
  int idx;
  int ix0;
  int k;
  bool guard1 = false;
  ix0 = mrows * ncols;
  guard1 = false;
  if (ix0 > 0) {
    for (idx = 0; idx < ncols; idx++) {
      int iy0;
      ix0 = 13 * idx;
      iy0 = 25 * idx;
      for (k = 0; k < mrows; k++) {
        obj->QR[iy0 + k] = A[ix0 + k];
      }
    }

    guard1 = true;
  } else if (ix0 == 0) {
    obj->mrows = mrows;
    obj->ncols = ncols;
    obj->minRowCol = 0;
  } else {
    guard1 = true;
  }

  if (guard1) {
    obj->usedPivoting = false;
    obj->mrows = mrows;
    obj->ncols = ncols;
    if (ncols < 400) {
      for (b_idx = 0; b_idx < ncols; b_idx++) {
        obj->jpvt[b_idx] = b_idx + 1;
      }
    } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

      for (b_idx = 0; b_idx < ncols; b_idx++) {
        obj->jpvt[b_idx] = b_idx + 1;
      }
    }

    if (mrows <= ncols) {
      ix0 = mrows;
    } else {
      ix0 = ncols;
    }

    obj->minRowCol = ix0;
    memset(&obj->tau[0], 0, 25U * sizeof(double));
    if (ix0 >= 1) {
      qrf(obj->QR, mrows, ncols, ix0, obj->tau);
    }
  }
}

/*
 * Arguments    : d_struct_T *objfun_workspace
 *                const double lb[12]
 *                const double ub[12]
 *                j_struct_T *obj
 * Return Type  : void
 */
static void factoryConstruct(d_struct_T *objfun_workspace, const double lb[12],
  const double ub[12], j_struct_T *obj)
{
  int i;
  bool bv[12];
  bool b;
  obj->objfun.workspace = *objfun_workspace;
  obj->f_1 = 0.0;
  obj->f_2 = 0.0;
  obj->nVar = 12;
  obj->mIneq = 0;
  obj->mEq = 0;
  obj->numEvals = 0;
  obj->SpecifyObjectiveGradient = true;
  obj->SpecifyConstraintGradient = false;
  obj->isEmptyNonlcon = true;
  obj->FiniteDifferenceType = 0;
  for (i = 0; i < 12; i++) {
    bv[i] = obj->hasUB[i];
  }

  b = false;
  i = 0;
  while ((!b) && (i + 1 <= 12)) {
    obj->hasLB[i] = ((!rtIsInf(lb[i])) && (!rtIsNaN(lb[i])));
    bv[i] = ((!rtIsInf(ub[i])) && (!rtIsNaN(ub[i])));
    if (obj->hasLB[i] || bv[i]) {
      b = true;
    }

    i++;
  }

  while (i + 1 <= 12) {
    obj->hasLB[i] = ((!rtIsInf(lb[i])) && (!rtIsNaN(lb[i])));
    bv[i] = ((!rtIsInf(ub[i])) && (!rtIsNaN(ub[i])));
    i++;
  }

  for (i = 0; i < 12; i++) {
    obj->hasUB[i] = bv[i];
  }

  obj->hasBounds = b;
}

/*
 * Arguments    : double workspace[325]
 *                double xCurrent[13]
 *                const k_struct_T *workingset
 *                f_struct_T *qrmanager
 * Return Type  : bool
 */
static bool feasibleX0ForWorkingSet(double workspace[325], double xCurrent[13],
  const k_struct_T *workingset, f_struct_T *qrmanager)
{
  double B[325];
  int ar;
  int b_i;
  int iAcol;
  int ia;
  int ic;
  int idx;
  int jBcol;
  int k;
  int mWConstr;
  int nVar;
  bool nonDegenerateWset;
  mWConstr = workingset->nActiveConstr;
  nVar = workingset->nVar;
  nonDegenerateWset = true;
  if (mWConstr != 0) {
    double c;
    int i;
    int i1;
    if (mWConstr < 400) {
      for (idx = 0; idx < mWConstr; idx++) {
        workspace[idx] = workingset->bwset[idx];
        workspace[idx + 25] = workingset->bwset[idx];
      }
    } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

      for (idx = 0; idx < mWConstr; idx++) {
        workspace[idx] = workingset->bwset[idx];
        workspace[idx + 25] = workingset->bwset[idx];
      }
    }

    if (mWConstr != 0) {
      i = 13 * (mWConstr - 1) + 1;
      for (iAcol = 1; iAcol <= i; iAcol += 13) {
        c = 0.0;
        i1 = (iAcol + nVar) - 1;
        for (ia = iAcol; ia <= i1; ia++) {
          c += workingset->ATwset[ia - 1] * xCurrent[ia - iAcol];
        }

        i1 = div_nde_s32_floor(iAcol - 1, 13);
        workspace[i1] += -c;
      }
    }

    if (mWConstr >= nVar) {
      qrmanager->usedPivoting = false;
      qrmanager->mrows = mWConstr;
      qrmanager->ncols = nVar;
      for (ia = 0; ia < nVar; ia++) {
        iAcol = 25 * ia;
        for (jBcol = 0; jBcol < mWConstr; jBcol++) {
          qrmanager->QR[jBcol + iAcol] = workingset->ATwset[ia + 13 * jBcol];
        }

        qrmanager->jpvt[ia] = ia + 1;
      }

      if (mWConstr <= nVar) {
        i = mWConstr;
      } else {
        i = nVar;
      }

      qrmanager->minRowCol = i;
      memset(&qrmanager->tau[0], 0, 25U * sizeof(double));
      if (i >= 1) {
        qrf(qrmanager->QR, mWConstr, nVar, i, qrmanager->tau);
      }

      computeQ_(qrmanager, mWConstr);
      memcpy(&B[0], &workspace[0], 325U * sizeof(double));
      for (k = 0; k <= 25; k += 25) {
        i = k + 1;
        i1 = k + nVar;
        if (i <= i1) {
          memset(&workspace[i + -1], 0, ((i1 - i) + 1) * sizeof(double));
        }
      }

      jBcol = -1;
      for (k = 0; k <= 25; k += 25) {
        ar = -1;
        i = k + 1;
        i1 = k + nVar;
        for (ic = i; ic <= i1; ic++) {
          c = 0.0;
          for (iAcol = 0; iAcol < mWConstr; iAcol++) {
            c += qrmanager->Q[(iAcol + ar) + 1] * B[(iAcol + jBcol) + 1];
          }

          workspace[ic - 1] += c;
          ar += 25;
        }

        jBcol += 25;
      }

      for (ar = 0; ar < 2; ar++) {
        jBcol = 25 * ar - 1;
        for (k = nVar; k >= 1; k--) {
          iAcol = 25 * (k - 1) - 1;
          i = k + jBcol;
          c = workspace[i];
          if (c != 0.0) {
            workspace[i] = c / qrmanager->QR[k + iAcol];
            for (b_i = 0; b_i <= k - 2; b_i++) {
              i1 = (b_i + jBcol) + 1;
              workspace[i1] -= workspace[i] * qrmanager->QR[(b_i + iAcol) + 1];
            }
          }
        }
      }
    } else {
      factorQR(qrmanager, workingset->ATwset, nVar, mWConstr);
      computeQ_(qrmanager, qrmanager->minRowCol);
      for (ar = 0; ar < 2; ar++) {
        jBcol = 25 * ar;
        for (b_i = 0; b_i < mWConstr; b_i++) {
          iAcol = 25 * b_i;
          ia = b_i + jBcol;
          c = workspace[ia];
          for (k = 0; k < b_i; k++) {
            c -= qrmanager->QR[k + iAcol] * workspace[k + jBcol];
          }

          workspace[ia] = c / qrmanager->QR[b_i + iAcol];
        }
      }

      memcpy(&B[0], &workspace[0], 325U * sizeof(double));
      for (k = 0; k <= 25; k += 25) {
        i = k + 1;
        i1 = k + nVar;
        if (i <= i1) {
          memset(&workspace[i + -1], 0, ((i1 - i) + 1) * sizeof(double));
        }
      }

      jBcol = 0;
      for (k = 0; k <= 25; k += 25) {
        ar = -1;
        i = jBcol + 1;
        i1 = jBcol + mWConstr;
        for (b_i = i; b_i <= i1; b_i++) {
          iAcol = k + 1;
          ia = k + nVar;
          for (ic = iAcol; ic <= ia; ic++) {
            workspace[ic - 1] += B[b_i - 1] * qrmanager->Q[(ar + ic) - k];
          }

          ar += 25;
        }

        jBcol += 25;
      }
    }

    iAcol = 0;
    int exitg1;
    do {
      exitg1 = 0;
      if (iAcol <= nVar - 1) {
        if (rtIsInf(workspace[iAcol]) || rtIsNaN(workspace[iAcol])) {
          nonDegenerateWset = false;
          exitg1 = 1;
        } else {
          c = workspace[iAcol + 25];
          if (rtIsInf(c) || rtIsNaN(c)) {
            nonDegenerateWset = false;
            exitg1 = 1;
          } else {
            iAcol++;
          }
        }
      } else {
        double constrViolation_basicX;
        iAcol = nVar - 1;
        for (k = 0; k <= iAcol; k++) {
          workspace[k] += xCurrent[k];
        }

        c = maxConstraintViolation(workingset, workspace, 1);
        constrViolation_basicX = maxConstraintViolation(workingset, workspace,
          26);
        if ((c <= 2.2204460492503131E-16) || (c < constrViolation_basicX)) {
          memcpy(&xCurrent[0], &workspace[0], nVar * sizeof(double));
        } else {
          memcpy(&xCurrent[0], &workspace[25], nVar * sizeof(double));
        }

        exitg1 = 1;
      }
    } while (exitg1 == 0);
  }

  return nonDegenerateWset;
}

/*
 * Arguments    : const double solution_xstar[13]
 *                const double solution_searchDir[13]
 *                int workingset_nVar
 *                const double workingset_lb[13]
 *                const double workingset_ub[13]
 *                const int workingset_indexLB[13]
 *                const int workingset_indexUB[13]
 *                const int workingset_sizes[5]
 *                const int workingset_isActiveIdx[6]
 *                const bool workingset_isActiveConstr[25]
 *                const int workingset_nWConstr[5]
 *                bool isPhaseOne
 *                double *alpha
 *                bool *newBlocking
 *                int *constrType
 *                int *constrIdx
 * Return Type  : void
 */
static void feasibleratiotest(const double solution_xstar[13], const double
  solution_searchDir[13], int workingset_nVar, const double workingset_lb[13],
  const double workingset_ub[13], const int workingset_indexLB[13], const int
  workingset_indexUB[13], const int workingset_sizes[5], const int
  workingset_isActiveIdx[6], const bool workingset_isActiveConstr[25], const int
  workingset_nWConstr[5], bool isPhaseOne, double *alpha, bool *newBlocking, int
  *constrType, int *constrIdx)
{
  double denomTol;
  double phaseOneCorrectionP;
  double phaseOneCorrectionX;
  double pk_corrected;
  double ratio;
  int i;
  int idx;
  int totalUB;
  totalUB = workingset_sizes[4];
  *alpha = 1.0E+30;
  *newBlocking = false;
  *constrType = 0;
  *constrIdx = 0;
  denomTol = 2.2204460492503131E-13 * b_xnrm2(workingset_nVar,
    solution_searchDir);
  if (workingset_nWConstr[3] < workingset_sizes[3]) {
    phaseOneCorrectionX = (double)isPhaseOne * solution_xstar[workingset_nVar -
      1];
    phaseOneCorrectionP = (double)isPhaseOne *
      solution_searchDir[workingset_nVar - 1];
    i = workingset_sizes[3];
    for (idx = 0; idx <= i - 2; idx++) {
      int i1;
      i1 = workingset_indexLB[idx];
      pk_corrected = -solution_searchDir[i1 - 1] - phaseOneCorrectionP;
      if ((pk_corrected > denomTol) && (!workingset_isActiveConstr
           [(workingset_isActiveIdx[3] + idx) - 1])) {
        ratio = (-solution_xstar[i1 - 1] - workingset_lb[i1 - 1]) -
          phaseOneCorrectionX;
        pk_corrected = fmin(fabs(ratio), 1.0E-6 - ratio) / pk_corrected;
        if (pk_corrected < *alpha) {
          *alpha = pk_corrected;
          *constrType = 4;
          *constrIdx = idx + 1;
          *newBlocking = true;
        }
      }
    }

    i = workingset_indexLB[workingset_sizes[3] - 1] - 1;
    pk_corrected = -solution_searchDir[i];
    if ((pk_corrected > denomTol) && (!workingset_isActiveConstr
         [(workingset_isActiveIdx[3] + workingset_sizes[3]) - 2])) {
      ratio = -solution_xstar[i] - workingset_lb[i];
      pk_corrected = fmin(fabs(ratio), 1.0E-6 - ratio) / pk_corrected;
      if (pk_corrected < *alpha) {
        *alpha = pk_corrected;
        *constrType = 4;
        *constrIdx = workingset_sizes[3];
        *newBlocking = true;
      }
    }
  }

  if (workingset_nWConstr[4] < workingset_sizes[4]) {
    phaseOneCorrectionX = (double)isPhaseOne * solution_xstar[workingset_nVar -
      1];
    phaseOneCorrectionP = (double)isPhaseOne *
      solution_searchDir[workingset_nVar - 1];
    for (idx = 0; idx < totalUB; idx++) {
      i = workingset_indexUB[idx];
      pk_corrected = solution_searchDir[i - 1] - phaseOneCorrectionP;
      if ((pk_corrected > denomTol) && (!workingset_isActiveConstr
           [(workingset_isActiveIdx[4] + idx) - 1])) {
        ratio = (solution_xstar[i - 1] - workingset_ub[i - 1]) -
          phaseOneCorrectionX;
        pk_corrected = fmin(fabs(ratio), 1.0E-6 - ratio) / pk_corrected;
        if (pk_corrected < *alpha) {
          *alpha = pk_corrected;
          *constrType = 5;
          *constrIdx = idx + 1;
          *newBlocking = true;
        }
      }
    }
  }

  if (!isPhaseOne) {
    if ((*newBlocking) && (*alpha > 1.0)) {
      *newBlocking = false;
    }

    *alpha = fmin(*alpha, 1.0);
  }
}

/*
 * Arguments    : d_struct_T *fun_workspace
 *                const double x0[12]
 *                const double lb[12]
 *                const double ub[12]
 *                double x[12]
 *                double *fval
 *                double *exitflag
 *                double *output_iterations
 *                double *output_funcCount
 *                char output_algorithm[3]
 *                double *output_constrviolation
 *                double *output_stepsize
 *                double *output_lssteplength
 *                double *output_firstorderopt
 * Return Type  : void
 */
static void fmincon(d_struct_T *fun_workspace, const double x0[12], const double
                    lb[12], const double ub[12], double x[12], double *fval,
                    double *exitflag, double *output_iterations, double
                    *output_funcCount, char output_algorithm[3], double
                    *output_constrviolation, double *output_stepsize, double
                    *output_lssteplength, double *output_firstorderopt)
{
  b_struct_T MeritFunction;
  d_struct_T FcnEvaluator_objfun_workspace;
  d_struct_T b_fun_workspace;
  e_struct_T memspace;
  f_struct_T QRManager;
  g_struct_T CholManager;
  i_struct_T TrialState;
  j_struct_T unusedExpr;
  k_struct_T WorkingSet;
  l_struct_T FcnEvaluator;
  struct_T QPObjective;
  double scale;
  double y;
  int b_i;
  int colOffsetATw;
  int i;
  int idx;
  int mFixed;
  int mLB;
  signed char b_obj_tmp[5];
  signed char obj_tmp[5];
  output_algorithm[0] = 's';
  output_algorithm[1] = 'q';
  output_algorithm[2] = 'p';
  TrialState.nVarMax = 13;
  TrialState.mNonlinIneq = 0;
  TrialState.mNonlinEq = 0;
  TrialState.mIneq = 0;
  TrialState.mEq = 0;
  TrialState.iNonIneq0 = 1;
  TrialState.iNonEq0 = 1;
  TrialState.sqpFval_old = 0.0;
  TrialState.sqpIterations = 0;
  TrialState.sqpExitFlag = 0;
  memset(&TrialState.lambdasqp[0], 0, 25U * sizeof(double));
  TrialState.steplength = 1.0;
  memset(&TrialState.delta_x[0], 0, 13U * sizeof(double));
  TrialState.fstar = 0.0;
  TrialState.firstorderopt = 0.0;
  memset(&TrialState.lambda[0], 0, 25U * sizeof(double));
  TrialState.state = 0;
  TrialState.maxConstr = 0.0;
  TrialState.iterations = 0;
  memcpy(&TrialState.xstarsqp[0], &x0[0], 12U * sizeof(double));
  FcnEvaluator_objfun_workspace = *fun_workspace;
  b_fun_workspace = *fun_workspace;
  factoryConstruct(&b_fun_workspace, lb, ub, &unusedExpr);
  WorkingSet.nVar = 12;
  WorkingSet.nVarOrig = 12;
  WorkingSet.nVarMax = 13;
  WorkingSet.ldA = 13;
  memset(&WorkingSet.lb[0], 0, 13U * sizeof(double));
  memset(&WorkingSet.ub[0], 0, 13U * sizeof(double));
  WorkingSet.mEqRemoved = 0;
  memset(&WorkingSet.ATwset[0], 0, 325U * sizeof(double));
  WorkingSet.nActiveConstr = 0;
  memset(&WorkingSet.bwset[0], 0, 25U * sizeof(double));
  memset(&WorkingSet.maxConstrWorkspace[0], 0, 25U * sizeof(double));
  memset(&WorkingSet.Wid[0], 0, 25U * sizeof(int));
  memset(&WorkingSet.Wlocalidx[0], 0, 25U * sizeof(int));
  for (i = 0; i < 25; i++) {
    WorkingSet.isActiveConstr[i] = false;
  }

  for (i = 0; i < 5; i++) {
    WorkingSet.nWConstr[i] = 0;
  }

  WorkingSet.probType = 3;
  WorkingSet.SLACK0 = 1.0E-5;
  for (i = 0; i < 13; i++) {
    WorkingSet.indexLB[i] = 0;
    WorkingSet.indexUB[i] = 0;
    WorkingSet.indexFixed[i] = 0;
  }

  mLB = 0;
  colOffsetATw = 0;
  mFixed = 0;
  for (i = 0; i < 12; i++) {
    bool guard1 = false;
    y = lb[i];
    guard1 = false;
    if ((!rtIsInf(y)) && (!rtIsNaN(y))) {
      if (fabs(y - ub[i]) < 1.0E-6) {
        mFixed++;
        WorkingSet.indexFixed[mFixed - 1] = i + 1;
      } else {
        mLB++;
        WorkingSet.indexLB[mLB - 1] = i + 1;
        guard1 = true;
      }
    } else {
      guard1 = true;
    }

    if (guard1) {
      y = ub[i];
      if ((!rtIsInf(y)) && (!rtIsNaN(y))) {
        colOffsetATw++;
        WorkingSet.indexUB[colOffsetATw - 1] = i + 1;
      }
    }
  }

  i = (mLB + colOffsetATw) + mFixed;
  WorkingSet.mConstr = i;
  WorkingSet.mConstrOrig = i;
  WorkingSet.mConstrMax = 25;
  obj_tmp[0] = (signed char)mFixed;
  obj_tmp[1] = 0;
  obj_tmp[2] = 0;
  obj_tmp[3] = (signed char)mLB;
  obj_tmp[4] = (signed char)colOffsetATw;
  b_obj_tmp[0] = (signed char)mFixed;
  b_obj_tmp[1] = 0;
  b_obj_tmp[2] = 0;
  b_obj_tmp[3] = (signed char)(mLB + 1);
  b_obj_tmp[4] = (signed char)colOffsetATw;
  WorkingSet.isActiveIdx[0] = 1;
  WorkingSet.isActiveIdx[1] = mFixed;
  WorkingSet.isActiveIdx[2] = 0;
  WorkingSet.isActiveIdx[3] = 0;
  WorkingSet.isActiveIdx[4] = mLB;
  WorkingSet.isActiveIdx[5] = colOffsetATw;
  for (i = 0; i < 5; i++) {
    signed char i1;
    signed char i2;
    i1 = obj_tmp[i];
    WorkingSet.sizes[i] = i1;
    WorkingSet.sizesNormal[i] = i1;
    i2 = b_obj_tmp[i];
    WorkingSet.sizesPhaseOne[i] = i2;
    WorkingSet.sizesRegularized[i] = i1;
    WorkingSet.sizesRegPhaseOne[i] = i2;
    WorkingSet.isActiveIdx[i + 1] += WorkingSet.isActiveIdx[i];
  }

  for (b_i = 0; b_i < 6; b_i++) {
    WorkingSet.isActiveIdxNormal[b_i] = WorkingSet.isActiveIdx[b_i];
  }

  WorkingSet.isActiveIdxPhaseOne[0] = 1;
  WorkingSet.isActiveIdxPhaseOne[1] = mFixed;
  WorkingSet.isActiveIdxPhaseOne[2] = 0;
  WorkingSet.isActiveIdxPhaseOne[3] = 0;
  WorkingSet.isActiveIdxPhaseOne[4] = mLB + 1;
  WorkingSet.isActiveIdxPhaseOne[5] = colOffsetATw;
  for (i = 0; i < 5; i++) {
    WorkingSet.isActiveIdxPhaseOne[i + 1] += WorkingSet.isActiveIdxPhaseOne[i];
  }

  for (b_i = 0; b_i < 6; b_i++) {
    WorkingSet.isActiveIdxRegularized[b_i] = WorkingSet.isActiveIdx[b_i];
    WorkingSet.isActiveIdxRegPhaseOne[b_i] = WorkingSet.isActiveIdxPhaseOne[b_i];
  }

  for (i = 0; i < mLB; i++) {
    b_i = WorkingSet.indexLB[i];
    TrialState.xstarsqp[b_i - 1] = fmax(TrialState.xstarsqp[b_i - 1], lb[b_i - 1]);
  }

  for (i = 0; i < colOffsetATw; i++) {
    b_i = WorkingSet.indexUB[i];
    TrialState.xstarsqp[b_i - 1] = fmin(TrialState.xstarsqp[b_i - 1], ub[b_i - 1]);
  }

  for (i = 0; i < mFixed; i++) {
    b_i = WorkingSet.indexFixed[i];
    TrialState.xstarsqp[b_i - 1] = ub[b_i - 1];
  }

  evalObjAndConstrAndDerivatives(&FcnEvaluator_objfun_workspace,
    TrialState.xstarsqp, TrialState.grad, &TrialState.sqpFval, &i);
  TrialState.FunctionEvaluations = 1;
  for (i = 0; i < mLB; i++) {
    WorkingSet.lb[WorkingSet.indexLB[i] - 1] = -lb[WorkingSet.indexLB[i] - 1] +
      x0[WorkingSet.indexLB[i] - 1];
  }

  for (i = 0; i < colOffsetATw; i++) {
    WorkingSet.ub[WorkingSet.indexUB[i] - 1] = ub[WorkingSet.indexUB[i] - 1] -
      x0[WorkingSet.indexUB[i] - 1];
  }

  for (i = 0; i < mFixed; i++) {
    y = ub[WorkingSet.indexFixed[i] - 1] - x0[WorkingSet.indexFixed[i] - 1];
    WorkingSet.ub[WorkingSet.indexFixed[i] - 1] = y;
    WorkingSet.bwset[i] = y;
  }

  setProblemType(&WorkingSet, 3);
  i = WorkingSet.isActiveIdx[2];
  if (26 - WorkingSet.isActiveIdx[2] < 400) {
    for (idx = i; idx < 26; idx++) {
      WorkingSet.isActiveConstr[idx - 1] = false;
    }
  } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

    for (idx = i; idx < 26; idx++) {
      WorkingSet.isActiveConstr[idx - 1] = false;
    }
  }

  WorkingSet.nWConstr[0] = WorkingSet.sizes[0];
  WorkingSet.nWConstr[1] = 0;
  WorkingSet.nWConstr[2] = 0;
  WorkingSet.nWConstr[3] = 0;
  WorkingSet.nWConstr[4] = 0;
  WorkingSet.nActiveConstr = WorkingSet.nWConstr[0];
  i = WorkingSet.sizes[0];
  for (mLB = 0; mLB < i; mLB++) {
    WorkingSet.Wid[mLB] = 1;
    WorkingSet.Wlocalidx[mLB] = mLB + 1;
    WorkingSet.isActiveConstr[mLB] = true;
    colOffsetATw = 13 * mLB;
    b_i = WorkingSet.indexFixed[mLB];
    if (b_i - 2 >= 0) {
      memset(&WorkingSet.ATwset[colOffsetATw], 0, (((b_i + colOffsetATw) -
               colOffsetATw) - 1) * sizeof(double));
    }

    WorkingSet.ATwset[(WorkingSet.indexFixed[mLB] + colOffsetATw) - 1] = 1.0;
    b_i = WorkingSet.indexFixed[mLB] + 1;
    mFixed = WorkingSet.nVar;
    if (b_i <= mFixed) {
      memset(&WorkingSet.ATwset[(b_i + colOffsetATw) + -1], 0, ((((mFixed +
                 colOffsetATw) - b_i) - colOffsetATw) + 1) * sizeof(double));
    }

    WorkingSet.bwset[mLB] = WorkingSet.ub[WorkingSet.indexFixed[mLB] - 1];
  }

  double Hessian[144];
  MeritFunction.initFval = TrialState.sqpFval;
  MeritFunction.penaltyParam = 1.0;
  MeritFunction.threshold = 0.0001;
  MeritFunction.nPenaltyDecreases = 0;
  MeritFunction.linearizedConstrViol = 0.0;
  MeritFunction.initConstrViolationEq = 0.0;
  MeritFunction.initConstrViolationIneq = 0.0;
  MeritFunction.phi = 0.0;
  MeritFunction.phiPrimePlus = 0.0;
  MeritFunction.phiFullStep = 0.0;
  MeritFunction.feasRelativeFactor = 0.0;
  MeritFunction.nlpPrimalFeasError = 0.0;
  MeritFunction.nlpDualFeasError = 0.0;
  MeritFunction.nlpComplError = 0.0;
  MeritFunction.firstOrderOpt = 0.0;
  MeritFunction.hasObjective = true;
  FcnEvaluator.objfun.workspace = FcnEvaluator_objfun_workspace;
  FcnEvaluator.nVar = 12;
  FcnEvaluator.mCineq = 0;
  FcnEvaluator.mCeq = 0;
  FcnEvaluator.NonFiniteSupport = true;
  FcnEvaluator.SpecifyObjectiveGradient = true;
  FcnEvaluator.SpecifyConstraintGradient = false;
  FcnEvaluator.ScaleProblem = false;
  b_driver(lb, ub, &TrialState, &MeritFunction, &FcnEvaluator, &memspace,
           &WorkingSet, Hessian, &QRManager, &CholManager, &QPObjective);
  *fval = TrialState.sqpFval;
  *exitflag = TrialState.sqpExitFlag;
  *output_iterations = TrialState.sqpIterations;
  *output_funcCount = TrialState.FunctionEvaluations;
  *output_constrviolation = MeritFunction.nlpPrimalFeasError;
  y = 0.0;
  scale = 3.3121686421112381E-170;
  for (i = 0; i < 12; i++) {
    double absxk;
    x[i] = TrialState.xstarsqp[i];
    absxk = fabs(TrialState.delta_x[i]);
    if (absxk > scale) {
      double t;
      t = scale / absxk;
      y = y * t * t + 1.0;
      scale = absxk;
    } else {
      double t;
      t = absxk / scale;
      y += t * t;
    }
  }

  *output_stepsize = scale * sqrt(y);
  *output_lssteplength = TrialState.steplength;
  *output_firstorderopt = MeritFunction.firstOrderOpt;
}

/*
 * Arguments    : g_struct_T *obj
 *                int NColsRemain
 * Return Type  : void
 */
static void fullColLDL2_(g_struct_T *obj, int NColsRemain)
{
  int ijA;
  int j;
  int jA;
  int k;
  for (k = 0; k < NColsRemain; k++) {
    double alpha1;
    double y;
    int LD_diagOffset;
    int i;
    int subMatrixDim;
    LD_diagOffset = 26 * k;
    alpha1 = -1.0 / obj->FMat[LD_diagOffset];
    subMatrixDim = (NColsRemain - k) - 2;
    for (jA = 0; jA <= subMatrixDim; jA++) {
      obj->workspace_ = obj->FMat[(LD_diagOffset + jA) + 1];
    }

    y = obj->workspace_;
    if (!(alpha1 == 0.0)) {
      jA = LD_diagOffset;
      for (j = 0; j <= subMatrixDim; j++) {
        if (y != 0.0) {
          double temp;
          int i1;
          temp = y * alpha1;
          i = jA + 27;
          i1 = subMatrixDim + jA;
          for (ijA = i; ijA <= i1 + 27; ijA++) {
            obj->FMat[ijA - 1] += obj->workspace_ * temp;
          }
        }

        jA += 25;
      }
    }

    for (jA = 0; jA <= subMatrixDim; jA++) {
      i = (LD_diagOffset + jA) + 1;
      obj->FMat[i] /= obj->FMat[LD_diagOffset];
    }
  }
}

/*
 * Arguments    : const double H[144]
 *                const double f[13]
 *                i_struct_T *solution
 *                e_struct_T *memspace
 *                k_struct_T *workingset
 *                f_struct_T *qrmanager
 *                g_struct_T *cholmanager
 *                struct_T *objective
 *                const char options_SolverName[7]
 *                double options_StepTolerance
 *                double options_ObjectiveLimit
 *                int runTimeOptions_MaxIterations
 * Return Type  : void
 */
static void iterate(const double H[144], const double f[13], i_struct_T
                    *solution, e_struct_T *memspace, k_struct_T *workingset,
                    f_struct_T *qrmanager, g_struct_T *cholmanager, struct_T
                    *objective, const char options_SolverName[7], double
                    options_StepTolerance, double options_ObjectiveLimit, int
                    runTimeOptions_MaxIterations)
{
  static const char b[7] = { 'f', 'm', 'i', 'n', 'c', 'o', 'n' };

  double c;
  double minLambda;
  double s;
  double temp_tmp;
  int TYPE;
  int activeSetChangeID;
  int globalActiveConstrIdx;
  int ia;
  int idx;
  int iyend;
  int n;
  int nActiveConstr;
  int nVar_tmp_tmp;
  bool subProblemChanged;
  bool updateFval;
  subProblemChanged = true;
  updateFval = true;
  activeSetChangeID = 0;
  TYPE = objective->objtype;
  nVar_tmp_tmp = workingset->nVar;
  globalActiveConstrIdx = 0;
  computeGrad_StoreHx(objective, H, f, solution->xstar);
  solution->fstar = computeFval_ReuseHx(objective, memspace->workspace_double, f,
    solution->xstar);
  if (solution->iterations < runTimeOptions_MaxIterations) {
    solution->state = -5;
  } else {
    solution->state = 0;
  }

  memset(&solution->lambda[0], 0, 25U * sizeof(double));
  int exitg1;
  do {
    exitg1 = 0;
    if (solution->state == -5) {
      int idxRotGCol;
      bool guard1 = false;
      bool guard2 = false;
      guard1 = false;
      guard2 = false;
      if (subProblemChanged) {
        switch (activeSetChangeID) {
         case 1:
          nActiveConstr = 13 * (workingset->nActiveConstr - 1);
          iyend = qrmanager->mrows;
          idxRotGCol = qrmanager->ncols + 1;
          if (iyend <= idxRotGCol) {
            idxRotGCol = iyend;
          }

          qrmanager->minRowCol = idxRotGCol;
          idxRotGCol = 25 * qrmanager->ncols;
          if (qrmanager->mrows != 0) {
            iyend = idxRotGCol + qrmanager->mrows;
            if (idxRotGCol + 1 <= iyend) {
              memset(&qrmanager->QR[idxRotGCol], 0, (iyend - idxRotGCol) *
                     sizeof(double));
            }

            n = 25 * (qrmanager->mrows - 1) + 1;
            for (idx = 1; idx <= n; idx += 25) {
              c = 0.0;
              iyend = (idx + qrmanager->mrows) - 1;
              for (ia = idx; ia <= iyend; ia++) {
                c += qrmanager->Q[ia - 1] * workingset->ATwset[(nActiveConstr +
                  ia) - idx];
              }

              iyend = idxRotGCol + div_nde_s32_floor(idx - 1, 25);
              qrmanager->QR[iyend] += c;
            }
          }

          qrmanager->ncols++;
          qrmanager->jpvt[qrmanager->ncols - 1] = qrmanager->ncols;
          for (idx = qrmanager->mrows - 2; idx + 2 > qrmanager->ncols; idx--) {
            idxRotGCol = 25 * (qrmanager->ncols - 1);
            n = (idx + idxRotGCol) + 1;
            temp_tmp = qrmanager->QR[n];
            xrotg(&qrmanager->QR[idx + idxRotGCol], &temp_tmp, &c, &s);
            qrmanager->QR[n] = temp_tmp;
            iyend = 25 * idx;
            n = qrmanager->mrows;
            if (qrmanager->mrows >= 1) {
              for (nActiveConstr = 0; nActiveConstr < n; nActiveConstr++) {
                idxRotGCol = iyend + nActiveConstr;
                minLambda = qrmanager->Q[idxRotGCol + 25];
                temp_tmp = qrmanager->Q[idxRotGCol];
                qrmanager->Q[idxRotGCol + 25] = c * minLambda - s * temp_tmp;
                qrmanager->Q[idxRotGCol] = c * temp_tmp + s * minLambda;
              }
            }
          }
          break;

         case -1:
          deleteColMoveEnd(qrmanager, globalActiveConstrIdx);
          break;

         default:
          factorQR(qrmanager, workingset->ATwset, nVar_tmp_tmp,
                   workingset->nActiveConstr);
          computeQ_(qrmanager, qrmanager->mrows);
          break;
        }

        iyend = memcmp(&options_SolverName[0], &b[0], 7);
        compute_deltax(H, solution, memspace, qrmanager, cholmanager, objective,
                       iyend == 0);
        if (solution->state != -5) {
          exitg1 = 1;
        } else if ((b_xnrm2(nVar_tmp_tmp, solution->searchDir) <
                    options_StepTolerance) || (workingset->nActiveConstr >=
                    nVar_tmp_tmp)) {
          guard2 = true;
        } else {
          feasibleratiotest(solution->xstar, solution->searchDir,
                            workingset->nVar, workingset->lb, workingset->ub,
                            workingset->indexLB, workingset->indexUB,
                            workingset->sizes, workingset->isActiveIdx,
                            workingset->isActiveConstr, workingset->nWConstr,
                            TYPE == 5, &minLambda, &updateFval, &n, &iyend);
          if (updateFval) {
            switch (n) {
             case 3:
              workingset->nWConstr[2]++;
              workingset->isActiveConstr[(workingset->isActiveIdx[2] + iyend) -
                2] = true;
              workingset->nActiveConstr++;
              workingset->Wid[workingset->nActiveConstr - 1] = 3;
              workingset->Wlocalidx[workingset->nActiveConstr - 1] = iyend;

              /* A check that is always false is detected at compile-time. Eliminating code that follows. */
              break;

             case 4:
              addBoundToActiveSetMatrix_(workingset, 4, iyend);
              break;

             default:
              addBoundToActiveSetMatrix_(workingset, 5, iyend);
              break;
            }

            activeSetChangeID = 1;
          } else {
            if (objective->objtype == 5) {
              if (b_xnrm2(objective->nvar, solution->searchDir) > 100.0 *
                  (double)objective->nvar * 1.4901161193847656E-8) {
                solution->state = 3;
              } else {
                solution->state = 4;
              }
            }

            subProblemChanged = false;
            if (workingset->nActiveConstr == 0) {
              solution->state = 1;
            }
          }

          if ((nVar_tmp_tmp >= 1) && (!(minLambda == 0.0))) {
            iyend = nVar_tmp_tmp - 1;
            for (nActiveConstr = 0; nActiveConstr <= iyend; nActiveConstr++) {
              solution->xstar[nActiveConstr] += minLambda * solution->
                searchDir[nActiveConstr];
            }
          }

          computeGrad_StoreHx(objective, H, f, solution->xstar);
          updateFval = true;
          guard1 = true;
        }
      } else {
        if (nVar_tmp_tmp - 1 >= 0) {
          memset(&solution->searchDir[0], 0, nVar_tmp_tmp * sizeof(double));
        }

        guard2 = true;
      }

      if (guard2) {
        nActiveConstr = qrmanager->ncols;
        if (qrmanager->ncols > 0) {
          bool b_guard1 = false;
          b_guard1 = false;
          if (objective->objtype != 4) {
            minLambda = 100.0 * (double)qrmanager->mrows *
              2.2204460492503131E-16;
            if ((qrmanager->mrows > 0) && (qrmanager->ncols > 0)) {
              updateFval = true;
            } else {
              updateFval = false;
            }

            if (updateFval) {
              bool b_guard2 = false;
              idx = qrmanager->ncols;
              b_guard2 = false;
              if (qrmanager->mrows < qrmanager->ncols) {
                iyend = qrmanager->mrows + 25 * (qrmanager->ncols - 1);
                while ((idx > qrmanager->mrows) && (fabs(qrmanager->QR[iyend - 1])
                        >= minLambda)) {
                  idx--;
                  iyend -= 25;
                }

                updateFval = (idx == qrmanager->mrows);
                if (updateFval) {
                  b_guard2 = true;
                }
              } else {
                b_guard2 = true;
              }

              if (b_guard2) {
                iyend = idx + 25 * (idx - 1);
                while ((idx >= 1) && (fabs(qrmanager->QR[iyend - 1]) >=
                                      minLambda)) {
                  idx--;
                  iyend -= 26;
                }

                updateFval = (idx == 0);
              }
            }

            if (!updateFval) {
              solution->state = -7;
            } else {
              b_guard1 = true;
            }
          } else {
            b_guard1 = true;
          }

          if (b_guard1) {
            n = qrmanager->ncols;
            xgemv(qrmanager->mrows, qrmanager->ncols, qrmanager->Q,
                  objective->grad, memspace->workspace_double);
            if (qrmanager->ncols != 0) {
              for (idx = n; idx >= 1; idx--) {
                iyend = (idx + (idx - 1) * 25) - 1;
                memspace->workspace_double[idx - 1] /= qrmanager->QR[iyend];
                for (ia = 0; ia <= idx - 2; ia++) {
                  idxRotGCol = (idx - ia) - 2;
                  memspace->workspace_double[idxRotGCol] -=
                    memspace->workspace_double[idx - 1] * qrmanager->QR[(iyend -
                    ia) - 1];
                }
              }
            }

            for (idx = 0; idx < nActiveConstr; idx++) {
              solution->lambda[idx] = -memspace->workspace_double[idx];
            }
          }
        }

        if ((solution->state != -7) || (workingset->nActiveConstr > nVar_tmp_tmp))
        {
          nActiveConstr = -1;
          minLambda = 0.0;
          n = (workingset->nWConstr[0] + workingset->nWConstr[1]) + 1;
          iyend = workingset->nActiveConstr;
          for (idx = n; idx <= iyend; idx++) {
            temp_tmp = solution->lambda[idx - 1];
            if (temp_tmp < minLambda) {
              minLambda = temp_tmp;
              nActiveConstr = idx - 1;
            }
          }

          if (nActiveConstr + 1 == 0) {
            solution->state = 1;
          } else {
            activeSetChangeID = -1;
            globalActiveConstrIdx = nActiveConstr + 1;
            subProblemChanged = true;
            iyend = workingset->Wid[nActiveConstr] - 1;
            workingset->isActiveConstr[(workingset->isActiveIdx[workingset->
              Wid[nActiveConstr] - 1] + workingset->Wlocalidx[nActiveConstr]) -
              2] = false;
            workingset->Wid[nActiveConstr] = workingset->Wid
              [workingset->nActiveConstr - 1];
            workingset->Wlocalidx[nActiveConstr] = workingset->
              Wlocalidx[workingset->nActiveConstr - 1];
            n = workingset->nVar;
            for (idx = 0; idx < n; idx++) {
              workingset->ATwset[idx + 13 * nActiveConstr] = workingset->
                ATwset[idx + 13 * (workingset->nActiveConstr - 1)];
            }

            workingset->bwset[nActiveConstr] = workingset->bwset
              [workingset->nActiveConstr - 1];
            workingset->nActiveConstr--;
            workingset->nWConstr[iyend]--;
            solution->lambda[nActiveConstr] = 0.0;
          }
        } else {
          nActiveConstr = workingset->nActiveConstr;
          activeSetChangeID = 0;
          globalActiveConstrIdx = workingset->nActiveConstr;
          subProblemChanged = true;
          iyend = workingset->nActiveConstr - 1;
          idxRotGCol = workingset->Wid[iyend] - 1;
          workingset->isActiveConstr[(workingset->isActiveIdx[idxRotGCol] +
            workingset->Wlocalidx[iyend]) - 2] = false;
          workingset->nActiveConstr--;
          workingset->nWConstr[idxRotGCol]--;
          solution->lambda[nActiveConstr - 1] = 0.0;
        }

        updateFval = false;
        guard1 = true;
      }

      if (guard1) {
        solution->iterations++;
        iyend = objective->nvar - 1;
        if ((solution->iterations >= runTimeOptions_MaxIterations) &&
            ((solution->state != 1) || (objective->objtype == 5))) {
          solution->state = 0;
        }

        if (solution->iterations - solution->iterations / 50 * 50 == 0) {
          solution->maxConstr = b_maxConstraintViolation(workingset,
            solution->xstar);
          minLambda = solution->maxConstr;
          if (objective->objtype == 5) {
            minLambda = solution->maxConstr - solution->xstar[iyend];
          }

          if (minLambda > 1.0E-6) {
            bool nonDegenerateWset;
            if (iyend >= 0) {
              memcpy(&solution->searchDir[0], &solution->xstar[0], (iyend + 1) *
                     sizeof(double));
            }

            nonDegenerateWset = feasibleX0ForWorkingSet
              (memspace->workspace_double, solution->searchDir, workingset,
               qrmanager);
            if ((!nonDegenerateWset) && (solution->state != 0)) {
              solution->state = -2;
            }

            activeSetChangeID = 0;
            minLambda = b_maxConstraintViolation(workingset, solution->searchDir);
            if (minLambda < solution->maxConstr) {
              if (iyend >= 0) {
                memcpy(&solution->xstar[0], &solution->searchDir[0], (iyend + 1)
                       * sizeof(double));
              }

              solution->maxConstr = minLambda;
            }
          }
        }

        if (updateFval && (options_ObjectiveLimit > rtMinusInf)) {
          solution->fstar = computeFval_ReuseHx(objective,
            memspace->workspace_double, f, solution->xstar);
          if ((solution->fstar < options_ObjectiveLimit) && ((solution->state !=
                0) || (objective->objtype != 5))) {
            solution->state = 2;
          }
        }
      }
    } else {
      if (!updateFval) {
        solution->fstar = computeFval_ReuseHx(objective,
          memspace->workspace_double, f, solution->xstar);
      }

      exitg1 = 1;
    }
  } while (exitg1 == 0);
}

/*
 * Arguments    : bool obj_hasLinear
 *                int obj_nvar
 *                double workspace[325]
 *                const double H[144]
 *                const double f[13]
 *                const double x[13]
 * Return Type  : void
 */
static void linearForm_(bool obj_hasLinear, int obj_nvar, double workspace[325],
  const double H[144], const double f[13], const double x[13])
{
  int i;
  int ia;
  int iac;
  int ix;
  ix = 0;
  if (obj_hasLinear) {
    if (obj_nvar < 400) {
      if (obj_nvar - 1 >= 0) {
        memcpy(&workspace[0], &f[0], obj_nvar * sizeof(double));
      }
    } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

      for (i = 0; i < obj_nvar; i++) {
        workspace[i] = f[i];
      }
    }

    ix = 1;
  }

  if (obj_nvar != 0) {
    int b_i;
    if (ix != 1) {
      if (obj_nvar < 400) {
        if (obj_nvar - 1 >= 0) {
          memset(&workspace[0], 0, obj_nvar * sizeof(double));
        }
      } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

        for (i = 0; i < obj_nvar; i++) {
          workspace[i] = 0.0;
        }
      }
    }

    ix = 0;
    b_i = obj_nvar * (obj_nvar - 1) + 1;
    for (iac = 1; obj_nvar < 0 ? iac >= b_i : iac <= b_i; iac += obj_nvar) {
      double c;
      int i1;
      c = 0.5 * x[ix];
      i1 = (iac + obj_nvar) - 1;
      for (ia = iac; ia <= i1; ia++) {
        int i2;
        i2 = ia - iac;
        workspace[i2] += H[ia - 1] * c;
      }

      ix++;
    }
  }
}

/*
 * Arguments    : const k_struct_T *obj
 *                const double x[325]
 *                int ix0
 * Return Type  : double
 */
static double maxConstraintViolation(const k_struct_T *obj, const double x[325],
  int ix0)
{
  double v;
  int idx;
  int mFixed;
  int mLB;
  int mUB;
  mLB = obj->sizes[3];
  mUB = obj->sizes[4];
  mFixed = obj->sizes[0];
  v = 0.0;
  if (obj->sizes[3] > 0) {
    for (idx = 0; idx < mLB; idx++) {
      int idxLB;
      idxLB = obj->indexLB[idx] - 1;
      v = fmax(v, -x[(ix0 + idxLB) - 1] - obj->lb[idxLB]);
    }
  }

  if (obj->sizes[4] > 0) {
    for (idx = 0; idx < mUB; idx++) {
      mLB = obj->indexUB[idx] - 1;
      v = fmax(v, x[(ix0 + mLB) - 1] - obj->ub[mLB]);
    }
  }

  if (obj->sizes[0] > 0) {
    for (idx = 0; idx < mFixed; idx++) {
      v = fmax(v, fabs(x[(ix0 + obj->indexFixed[idx]) - 2] - obj->ub
                       [obj->indexFixed[idx] - 1]));
    }
  }

  return v;
}

/*
 * Arguments    : const double x[12]
 *                double *ex
 *                int *idx
 * Return Type  : void
 */
static void minimum(const double x[12], double *ex, int *idx)
{
  int k;
  if (!rtIsNaN(x[0])) {
    *idx = 1;
  } else {
    bool exitg1;
    *idx = 0;
    k = 2;
    exitg1 = false;
    while ((!exitg1) && (k < 13)) {
      if (!rtIsNaN(x[k - 1])) {
        *idx = k;
        exitg1 = true;
      } else {
        k++;
      }
    }
  }

  if (*idx == 0) {
    *ex = x[0];
    *idx = 1;
  } else {
    int i;
    *ex = x[*idx - 1];
    i = *idx + 1;
    for (k = i; k < 13; k++) {
      double d;
      d = x[k - 1];
      if (*ex > d) {
        *ex = d;
        *idx = k;
      }
    }
  }
}

/*
 * Arguments    : const double A_data[]
 *                const int A_size[2]
 *                const double B[18]
 *                double Y_data[]
 *                int *Y_size
 * Return Type  : void
 */
static void mldivide(const double A_data[], const int A_size[2], const double B
                     [18], double Y_data[], int *Y_size)
{
  double b_A_data[216];
  double b_B[18];
  double tau_data[12];
  double vn1_data[12];
  double vn2_data[12];
  double work_data[12];
  double absxk;
  double scale;
  double t;
  double y;
  int coltop;
  int i;
  int ix;
  int ix0;
  int j;
  int k;
  int kend;
  int lastc;
  int pvt;
  signed char jpvt_data[12];
  if (A_size[1] == 0) {
    *Y_size = 0;
  } else {
    double smax;
    int b_i;
    int iy;
    int n;
    *Y_size = A_size[1];
    ix = 18 * A_size[1];
    memcpy(&b_A_data[0], &A_data[0], ix * sizeof(double));
    n = A_size[1];
    ix = A_size[1];
    memset(&tau_data[0], 0, ix * sizeof(double));
    ix = A_size[1];
    memset(&jpvt_data[0], 0, ix * sizeof(signed char));
    for (lastc = 0; lastc < n; lastc++) {
      jpvt_data[lastc] = (signed char)(lastc + 1);
    }

    ix = A_size[1];
    memset(&work_data[0], 0, ix * sizeof(double));
    ix = A_size[1];
    memset(&vn1_data[0], 0, ix * sizeof(double));
    ix = A_size[1];
    memset(&vn2_data[0], 0, ix * sizeof(double));

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4) \
 private(t,absxk,k,kend,scale,y,ix0)

    for (j = 0; j < n; j++) {
      ix0 = j * 18;
      y = 0.0;
      scale = 3.3121686421112381E-170;
      kend = ix0 + 18;
      for (k = ix0 + 1; k <= kend; k++) {
        absxk = fabs(A_data[k - 1]);
        if (absxk > scale) {
          t = scale / absxk;
          y = y * t * t + 1.0;
          scale = absxk;
        } else {
          t = absxk / scale;
          y += t * t;
        }
      }

      t = scale * sqrt(y);
      vn1_data[j] = t;
      vn2_data[j] = t;
    }

    for (i = 0; i < n; i++) {
      double atmp;
      double d;
      double s;
      int ii;
      int ip1;
      int nmi;
      ip1 = i + 2;
      ii = i * 18 + i;
      nmi = n - i;
      if (nmi < 1) {
        ix = -1;
      } else {
        ix = 0;
        if (nmi > 1) {
          smax = fabs(vn1_data[i]);
          for (lastc = 2; lastc <= nmi; lastc++) {
            s = fabs(vn1_data[(i + lastc) - 1]);
            if (s > smax) {
              ix = lastc - 1;
              smax = s;
            }
          }
        }
      }

      pvt = i + ix;
      if (pvt != i) {
        ix = pvt * 18;
        iy = i * 18;
        for (lastc = 0; lastc < 18; lastc++) {
          coltop = ix + lastc;
          smax = b_A_data[coltop];
          b_i = iy + lastc;
          b_A_data[coltop] = b_A_data[b_i];
          b_A_data[b_i] = smax;
        }

        ix = jpvt_data[pvt];
        jpvt_data[pvt] = jpvt_data[i];
        jpvt_data[i] = (signed char)ix;
        vn1_data[pvt] = vn1_data[i];
        vn2_data[pvt] = vn2_data[i];
      }

      atmp = b_A_data[ii];
      coltop = ii + 2;
      tau_data[i] = 0.0;
      smax = c_xnrm2(17 - i, b_A_data, ii + 2);
      if (smax != 0.0) {
        d = b_A_data[ii];
        s = rt_hypotd_snf(d, smax);
        if (d >= 0.0) {
          s = -s;
        }

        if (fabs(s) < 1.0020841800044864E-292) {
          ix = 0;
          b_i = (ii - i) + 18;
          do {
            ix++;
            for (lastc = coltop; lastc <= b_i; lastc++) {
              b_A_data[lastc - 1] *= 9.9792015476736E+291;
            }

            s *= 9.9792015476736E+291;
            atmp *= 9.9792015476736E+291;
          } while ((fabs(s) < 1.0020841800044864E-292) && (ix < 20));

          s = rt_hypotd_snf(atmp, c_xnrm2(17 - i, b_A_data, ii + 2));
          if (atmp >= 0.0) {
            s = -s;
          }

          tau_data[i] = (s - atmp) / s;
          smax = 1.0 / (atmp - s);
          for (lastc = coltop; lastc <= b_i; lastc++) {
            b_A_data[lastc - 1] *= smax;
          }

          for (lastc = 0; lastc < ix; lastc++) {
            s *= 1.0020841800044864E-292;
          }

          atmp = s;
        } else {
          tau_data[i] = (s - d) / s;
          smax = 1.0 / (d - s);
          b_i = (ii - i) + 18;
          for (lastc = coltop; lastc <= b_i; lastc++) {
            b_A_data[lastc - 1] *= smax;
          }

          atmp = s;
        }
      }

      b_A_data[ii] = atmp;
      if (i + 1 < n) {
        int lastv;
        b_A_data[ii] = 1.0;
        pvt = ii + 19;
        if (tau_data[i] != 0.0) {
          bool exitg2;
          lastv = 18 - i;
          iy = (ii - i) + 17;
          while ((lastv > 0) && (b_A_data[iy] == 0.0)) {
            lastv--;
            iy--;
          }

          lastc = nmi - 2;
          exitg2 = false;
          while ((!exitg2) && (lastc + 1 > 0)) {
            int exitg1;
            coltop = (ii + lastc * 18) + 18;
            ix = coltop;
            do {
              exitg1 = 0;
              if (ix + 1 <= coltop + lastv) {
                if (b_A_data[ix] != 0.0) {
                  exitg1 = 1;
                } else {
                  ix++;
                }
              } else {
                lastc--;
                exitg1 = 2;
              }
            } while (exitg1 == 0);

            if (exitg1 == 1) {
              exitg2 = true;
            }
          }
        } else {
          lastv = 0;
          lastc = -1;
        }

        if (lastv > 0) {
          if (lastc + 1 != 0) {
            if (lastc >= 0) {
              memset(&work_data[0], 0, (lastc + 1) * sizeof(double));
            }

            b_i = (ii + 18 * lastc) + 19;
            for (coltop = pvt; coltop <= b_i; coltop += 18) {
              smax = 0.0;
              iy = (coltop + lastv) - 1;
              for (ix = coltop; ix <= iy; ix++) {
                smax += b_A_data[ix - 1] * b_A_data[(ii + ix) - coltop];
              }

              ix = div_nde_s32_floor((coltop - ii) - 19, 18);
              work_data[ix] += smax;
            }
          }

          if (!(-tau_data[i] == 0.0)) {
            coltop = ii;
            for (pvt = 0; pvt <= lastc; pvt++) {
              d = work_data[pvt];
              if (d != 0.0) {
                smax = d * -tau_data[i];
                b_i = coltop + 19;
                iy = lastv + coltop;
                for (ix = b_i; ix <= iy + 18; ix++) {
                  b_A_data[ix - 1] += b_A_data[((ii + ix) - coltop) - 19] * smax;
                }
              }

              coltop += 18;
            }
          }
        }

        b_A_data[ii] = atmp;
      }

      for (pvt = ip1; pvt <= n; pvt++) {
        ix = i + (pvt - 1) * 18;
        d = vn1_data[pvt - 1];
        if (d != 0.0) {
          smax = fabs(b_A_data[ix]) / d;
          smax = 1.0 - smax * smax;
          if (smax < 0.0) {
            smax = 0.0;
          }

          s = d / vn2_data[pvt - 1];
          s = smax * (s * s);
          if (s <= 1.4901161193847656E-8) {
            d = c_xnrm2(17 - i, b_A_data, ix + 2);
            vn1_data[pvt - 1] = d;
            vn2_data[pvt - 1] = d;
          } else {
            vn1_data[pvt - 1] = d * sqrt(smax);
          }
        }
      }
    }

    coltop = 0;
    smax = 3.9968028886505635E-14 * fabs(b_A_data[0]);
    while ((coltop < *Y_size) && (!(fabs(b_A_data[coltop + 18 * coltop]) <= smax)))
    {
      coltop++;
    }

    memcpy(&b_B[0], &B[0], 18U * sizeof(double));
    memset(&Y_data[0], 0, *Y_size * sizeof(double));
    for (pvt = 0; pvt < *Y_size; pvt++) {
      if (tau_data[pvt] != 0.0) {
        smax = b_B[pvt];
        b_i = pvt + 2;
        for (i = b_i; i < 19; i++) {
          smax += b_A_data[(i + 18 * pvt) - 1] * b_B[i - 1];
        }

        smax *= tau_data[pvt];
        if (smax != 0.0) {
          b_B[pvt] -= smax;
          for (i = b_i; i < 19; i++) {
            b_B[i - 1] -= b_A_data[(i + 18 * pvt) - 1] * smax;
          }
        }
      }
    }

    for (i = 0; i < coltop; i++) {
      Y_data[jpvt_data[i] - 1] = b_B[i];
    }

    for (pvt = coltop; pvt >= 1; pvt--) {
      signed char i1;
      i1 = jpvt_data[pvt - 1];
      ix = 18 * (pvt - 1);
      Y_data[i1 - 1] /= b_A_data[(pvt + ix) - 1];
      for (i = 0; i <= pvt - 2; i++) {
        iy = jpvt_data[i] - 1;
        Y_data[iy] -= Y_data[jpvt_data[pvt - 1] - 1] * b_A_data[i + ix];
      }
    }
  }
}

/*
 * Arguments    : double A[625]
 *                int m
 *                int n
 *                int nfxd
 *                double tau[25]
 * Return Type  : void
 */
static void qrf(double A[625], int m, int n, int nfxd, double tau[25])
{
  double work[25];
  double atmp;
  int i;
  memset(&tau[0], 0, 25U * sizeof(double));
  memset(&work[0], 0, 25U * sizeof(double));
  for (i = 0; i < nfxd; i++) {
    double d;
    int ii;
    int mmi;
    ii = i * 25 + i;
    mmi = m - i;
    if (i + 1 < m) {
      atmp = A[ii];
      d = xzlarfg(mmi, &atmp, A, ii + 2);
      tau[i] = d;
      A[ii] = atmp;
    } else {
      d = 0.0;
      tau[i] = 0.0;
    }

    if (i + 1 < n) {
      atmp = A[ii];
      A[ii] = 1.0;
      xzlarf(mmi, (n - i) - 1, ii + 1, d, A, ii + 26, work);
      A[ii] = atmp;
    }
  }
}

/*
 * Arguments    : double u0
 *                double u1
 * Return Type  : double
 */
static double rt_hypotd_snf(double u0, double u1)
{
  double a;
  double y;
  a = fabs(u0);
  y = fabs(u1);
  if (a < y) {
    a /= y;
    y *= sqrt(a * a + 1.0);
  } else if (a > y) {
    y /= a;
    y = a * sqrt(y * y + 1.0);
  } else if (!rtIsNaN(y)) {
    y = a * 1.4142135623730951;
  }

  return y;
}

/*
 * Arguments    : k_struct_T *obj
 *                int PROBLEM_TYPE
 * Return Type  : void
 */
static void setProblemType(k_struct_T *obj, int PROBLEM_TYPE)
{
  int i;
  int idx;
  int idxStartIneq;
  int idx_col;
  switch (PROBLEM_TYPE) {
   case 3:
    obj->nVar = 12;
    obj->mConstr = obj->mConstrOrig;
    if (obj->nWConstr[4] > 0) {
      i = obj->sizesNormal[4];
      for (idxStartIneq = 0; idxStartIneq < i; idxStartIneq++) {
        obj->isActiveConstr[(obj->isActiveIdxNormal[4] + idxStartIneq) - 1] =
          obj->isActiveConstr[(obj->isActiveIdx[4] + idxStartIneq) - 1];
      }
    }

    for (i = 0; i < 5; i++) {
      obj->sizes[i] = obj->sizesNormal[i];
    }

    for (i = 0; i < 6; i++) {
      obj->isActiveIdx[i] = obj->isActiveIdxNormal[i];
    }
    break;

   case 1:
    obj->nVar = 13;
    obj->mConstr = obj->mConstrOrig + 1;
    for (i = 0; i < 5; i++) {
      obj->sizes[i] = obj->sizesPhaseOne[i];
    }

    for (i = 0; i < 6; i++) {
      obj->isActiveIdx[i] = obj->isActiveIdxPhaseOne[i];
    }

    i = obj->sizes[0];
    if (i < 400) {
      for (idx = 0; idx < i; idx++) {
        obj->ATwset[13 * idx + 12] = 0.0;
      }
    } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

      for (idx = 0; idx < i; idx++) {
        obj->ATwset[13 * idx + 12] = 0.0;
      }
    }

    obj->indexLB[obj->sizes[3] - 1] = 13;
    obj->lb[12] = 1.0E-5;
    idxStartIneq = obj->isActiveIdx[2];
    i = obj->nActiveConstr;
    if ((i - idxStartIneq) + 1 < 400) {
      for (idx = idxStartIneq; idx <= i; idx++) {
        obj->ATwset[13 * (idx - 1) + 12] = -1.0;
      }
    } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

      for (idx = idxStartIneq; idx <= i; idx++) {
        obj->ATwset[13 * (idx - 1) + 12] = -1.0;
      }
    }

    if (obj->nWConstr[4] > 0) {
      i = obj->sizesNormal[4];
      for (idxStartIneq = 0; idxStartIneq <= i; idxStartIneq++) {
        obj->isActiveConstr[(obj->isActiveIdx[4] + idxStartIneq) - 1] = false;
      }
    }

    obj->isActiveConstr[obj->isActiveIdx[4] - 2] = false;
    break;

   case 2:
    {
      obj->nVar = 12;
      obj->mConstr = 24;
      for (i = 0; i < 5; i++) {
        obj->sizes[i] = obj->sizesRegularized[i];
      }

      if (obj->probType != 4) {
        int i1;
        int idx_lb;
        idx_lb = 12;
        i = obj->sizesNormal[3] + 1;
        i1 = obj->sizesRegularized[3];
        for (idxStartIneq = i; idxStartIneq <= i1; idxStartIneq++) {
          idx_lb++;
          obj->indexLB[idxStartIneq - 1] = idx_lb;
        }

        if (obj->nWConstr[4] > 0) {
          i = obj->sizesRegularized[4];
          for (idxStartIneq = 0; idxStartIneq < i; idxStartIneq++) {
            obj->isActiveConstr[obj->isActiveIdxRegularized[4] + idxStartIneq] =
              obj->isActiveConstr[(obj->isActiveIdx[4] + idxStartIneq) - 1];
          }
        }

        i = obj->isActiveIdx[4];
        i1 = obj->isActiveIdxRegularized[4] - 1;
        if ((i1 - i) + 1 < 400) {
          if (i <= i1) {
            memset(&obj->isActiveConstr[i + -1], 0, ((i1 - i) + 1) * sizeof(bool));
          }
        } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

          for (idx = i; idx <= i1; idx++) {
            obj->isActiveConstr[idx - 1] = false;
          }
        }

        idxStartIneq = obj->isActiveIdx[2];
        i = obj->nActiveConstr;
        for (idx_col = idxStartIneq; idx_col <= i; idx_col++) {
          idx_lb = 13 * (idx_col - 1) - 1;
          if (obj->Wid[idx_col - 1] == 3) {
            i1 = obj->Wlocalidx[idx_col - 1] + 11;
            if (i1 >= 13) {
              memset(&obj->ATwset[idx_lb + 13], 0, (((i1 + idx_lb) - idx_lb) -
                      12) * sizeof(double));
            }

            obj->ATwset[(obj->Wlocalidx[idx_col - 1] + idx_lb) + 12] = -1.0;
            i1 = obj->Wlocalidx[idx_col - 1] + 13;
            if (i1 <= 12) {
              memset(&obj->ATwset[i1 + idx_lb], 0, (((idx_lb - i1) - idx_lb) +
                      13) * sizeof(double));
            }
          }
        }
      }

      for (i = 0; i < 6; i++) {
        obj->isActiveIdx[i] = obj->isActiveIdxRegularized[i];
      }
    }
    break;

   default:
    obj->nVar = 13;
    obj->mConstr = 25;
    for (i = 0; i < 5; i++) {
      obj->sizes[i] = obj->sizesRegPhaseOne[i];
    }

    for (i = 0; i < 6; i++) {
      obj->isActiveIdx[i] = obj->isActiveIdxRegPhaseOne[i];
    }

    i = obj->sizes[0];
    if (i < 400) {
      for (idx = 0; idx < i; idx++) {
        obj->ATwset[13 * idx + 12] = 0.0;
      }
    } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

      for (idx = 0; idx < i; idx++) {
        obj->ATwset[13 * idx + 12] = 0.0;
      }
    }

    obj->indexLB[obj->sizes[3] - 1] = 13;
    obj->lb[12] = 1.0E-5;
    idxStartIneq = obj->isActiveIdx[2];
    i = obj->nActiveConstr;
    if ((i - idxStartIneq) + 1 < 400) {
      for (idx = idxStartIneq; idx <= i; idx++) {
        obj->ATwset[13 * (idx - 1) + 12] = -1.0;
      }
    } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

      for (idx = idxStartIneq; idx <= i; idx++) {
        obj->ATwset[13 * (idx - 1) + 12] = -1.0;
      }
    }

    if (obj->nWConstr[4] > 0) {
      i = obj->sizesNormal[4];
      for (idxStartIneq = 0; idxStartIneq <= i; idxStartIneq++) {
        obj->isActiveConstr[(obj->isActiveIdx[4] + idxStartIneq) - 1] = false;
      }
    }

    obj->isActiveConstr[obj->isActiveIdx[4] - 2] = false;
    break;
  }

  obj->probType = PROBLEM_TYPE;
}

/*
 * Arguments    : const g_struct_T *obj
 *                double rhs[13]
 * Return Type  : void
 */
static void solve(const g_struct_T *obj, double rhs[13])
{
  int i;
  int j;
  int jA;
  int n_tmp;
  n_tmp = obj->ndims;
  if (obj->ndims != 0) {
    for (j = 0; j < n_tmp; j++) {
      double temp;
      jA = j * 25;
      temp = rhs[j];
      for (i = 0; i < j; i++) {
        temp -= obj->FMat[jA + i] * rhs[i];
      }

      rhs[j] = temp / obj->FMat[jA + j];
    }
  }

  if (obj->ndims != 0) {
    for (j = n_tmp; j >= 1; j--) {
      jA = (j + (j - 1) * 25) - 1;
      rhs[j - 1] /= obj->FMat[jA];
      for (i = 0; i <= j - 2; i++) {
        int ix;
        ix = (j - i) - 2;
        rhs[ix] -= rhs[j - 1] * obj->FMat[(jA - i) - 1];
      }
    }
  }
}

/*
 * Arguments    : double lambda[25]
 *                int WorkingSet_nActiveConstr
 *                const int WorkingSet_sizes[5]
 *                const int WorkingSet_isActiveIdx[6]
 *                const int WorkingSet_Wid[25]
 *                const int WorkingSet_Wlocalidx[25]
 *                double workspace[325]
 * Return Type  : void
 */
static void sortLambdaQP(double lambda[25], int WorkingSet_nActiveConstr, const
  int WorkingSet_sizes[5], const int WorkingSet_isActiveIdx[6], const int
  WorkingSet_Wid[25], const int WorkingSet_Wlocalidx[25], double workspace[325])
{
  int k;
  if (WorkingSet_nActiveConstr != 0) {
    int idx;
    int idxOffset;
    int mAll;
    mAll = ((WorkingSet_sizes[0] + WorkingSet_sizes[3]) + WorkingSet_sizes[4]) -
      1;
    if (mAll + 1 < 400) {
      for (k = 0; k <= mAll; k++) {
        workspace[k] = lambda[k];
        lambda[k] = 0.0;
      }
    } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

      for (k = 0; k <= mAll; k++) {
        workspace[k] = lambda[k];
        lambda[k] = 0.0;
      }
    }

    mAll = 0;
    idx = 0;
    while ((idx + 1 <= WorkingSet_nActiveConstr) && (WorkingSet_Wid[idx] <= 2))
    {
      if (WorkingSet_Wid[idx] == 1) {
        idxOffset = 1;
      } else {
        idxOffset = WorkingSet_isActiveIdx[1];
      }

      lambda[(idxOffset + WorkingSet_Wlocalidx[idx]) - 2] = workspace[mAll];
      mAll++;
      idx++;
    }

    while (idx + 1 <= WorkingSet_nActiveConstr) {
      switch (WorkingSet_Wid[idx]) {
       case 3:
        idxOffset = WorkingSet_isActiveIdx[2];
        break;

       case 4:
        idxOffset = WorkingSet_isActiveIdx[3];
        break;

       default:
        idxOffset = WorkingSet_isActiveIdx[4];
        break;
      }

      lambda[(idxOffset + WorkingSet_Wlocalidx[idx]) - 2] = workspace[mAll];
      mAll++;
      idx++;
    }
  }
}

/*
 * Arguments    : int *STEP_TYPE
 *                double Hessian[144]
 *                const double lb[12]
 *                const double ub[12]
 *                i_struct_T *TrialState
 *                b_struct_T *MeritFunction
 *                e_struct_T *memspace
 *                k_struct_T *WorkingSet
 *                f_struct_T *QRManager
 *                g_struct_T *CholManager
 *                struct_T *QPObjective
 *                h_struct_T *qpoptions
 * Return Type  : bool
 */
static bool step(int *STEP_TYPE, double Hessian[144], const double lb[12], const
                 double ub[12], i_struct_T *TrialState, b_struct_T
                 *MeritFunction, e_struct_T *memspace, k_struct_T *WorkingSet,
                 f_struct_T *QRManager, g_struct_T *CholManager, struct_T
                 *QPObjective, h_struct_T *qpoptions)
{
  h_struct_T b_qpoptions;
  double dv[13];
  double oldDirIdx;
  double s;
  int idx;
  int idxStartIneq;
  int k;
  int mUB;
  int nVar;
  int nVarOrig;
  bool checkBoundViolation;
  bool stepSuccess;
  stepSuccess = true;
  checkBoundViolation = true;
  nVar = WorkingSet->nVar - 1;
  if (*STEP_TYPE != 3) {
    memcpy(&TrialState->xstar[0], &TrialState->xstarsqp[0], (nVar + 1) * sizeof
           (double));
  } else if (nVar + 1 < 400) {
    if (nVar >= 0) {
      memcpy(&TrialState->searchDir[0], &TrialState->xstar[0], (nVar + 1) *
             sizeof(double));
    }
  } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

    for (k = 0; k <= nVar; k++) {
      TrialState->searchDir[k] = TrialState->xstar[k];
    }
  }

  int exitg1;
  bool guard1 = false;
  do {
    exitg1 = 0;
    guard1 = false;
    switch (*STEP_TYPE) {
     case 1:
      b_qpoptions = *qpoptions;
      driver(Hessian, TrialState->grad, TrialState, memspace, WorkingSet,
             QRManager, CholManager, QPObjective, &b_qpoptions,
             qpoptions->MaxIterations);
      if (TrialState->state > 0) {
        MeritFunction->phi = TrialState->sqpFval;
        MeritFunction->linearizedConstrViol = 0.0;
        MeritFunction->penaltyParam = 1.0;
        MeritFunction->phiPrimePlus = fmin(TrialState->fstar, 0.0);
      }

      sortLambdaQP(TrialState->lambda, WorkingSet->nActiveConstr,
                   WorkingSet->sizes, WorkingSet->isActiveIdx, WorkingSet->Wid,
                   WorkingSet->Wlocalidx, memspace->workspace_double);
      if ((TrialState->state <= 0) && (TrialState->state != -6)) {
        *STEP_TYPE = 2;
      } else {
        if (nVar >= 0) {
          memcpy(&TrialState->delta_x[0], &TrialState->xstar[0], (nVar + 1) *
                 sizeof(double));
        }

        guard1 = true;
      }
      break;

     case 2:
      {
        double beta;
        idxStartIneq = (WorkingSet->nWConstr[0] + WorkingSet->nWConstr[1]) + 1;
        mUB = WorkingSet->nActiveConstr;
        for (nVarOrig = idxStartIneq; nVarOrig <= mUB; nVarOrig++) {
          WorkingSet->isActiveConstr[(WorkingSet->isActiveIdx[WorkingSet->
            Wid[nVarOrig - 1] - 1] + WorkingSet->Wlocalidx[nVarOrig - 1]) - 2] =
            false;
        }

        WorkingSet->nWConstr[2] = 0;
        WorkingSet->nWConstr[3] = 0;
        WorkingSet->nWConstr[4] = 0;
        WorkingSet->nActiveConstr = WorkingSet->nWConstr[0] +
          WorkingSet->nWConstr[1];
        memcpy(&dv[0], &TrialState->xstar[0], 13U * sizeof(double));
        idxStartIneq = WorkingSet->sizes[3];
        mUB = WorkingSet->sizes[4];
        for (idx = 0; idx < idxStartIneq; idx++) {
          oldDirIdx = WorkingSet->lb[WorkingSet->indexLB[idx] - 1];
          if (-dv[WorkingSet->indexLB[idx] - 1] > oldDirIdx) {
            if (rtIsInf(ub[WorkingSet->indexLB[idx] - 1])) {
              dv[WorkingSet->indexLB[idx] - 1] = -oldDirIdx + fabs(oldDirIdx);
            } else {
              dv[WorkingSet->indexLB[idx] - 1] = (WorkingSet->ub
                [WorkingSet->indexLB[idx] - 1] - oldDirIdx) / 2.0;
            }
          }
        }

        for (idx = 0; idx < mUB; idx++) {
          oldDirIdx = WorkingSet->ub[WorkingSet->indexUB[idx] - 1];
          if (dv[WorkingSet->indexUB[idx] - 1] > oldDirIdx) {
            if (rtIsInf(lb[WorkingSet->indexUB[idx] - 1])) {
              dv[WorkingSet->indexUB[idx] - 1] = oldDirIdx - fabs(oldDirIdx);
            } else {
              dv[WorkingSet->indexUB[idx] - 1] = (oldDirIdx - WorkingSet->
                lb[WorkingSet->indexUB[idx] - 1]) / 2.0;
            }
          }
        }

        memcpy(&TrialState->xstar[0], &dv[0], 13U * sizeof(double));
        nVarOrig = WorkingSet->nVar;
        beta = 0.0;
        for (idx = 0; idx < nVarOrig; idx++) {
          beta += Hessian[idx + 12 * idx];
        }

        beta /= (double)WorkingSet->nVar;
        if (TrialState->sqpIterations <= 1) {
          mUB = QPObjective->nvar;
          if (QPObjective->nvar < 1) {
            idxStartIneq = 0;
          } else {
            idxStartIneq = 1;
            if (QPObjective->nvar > 1) {
              oldDirIdx = fabs(TrialState->grad[0]);
              for (idx = 2; idx <= mUB; idx++) {
                s = fabs(TrialState->grad[idx - 1]);
                if (s > oldDirIdx) {
                  idxStartIneq = idx;
                  oldDirIdx = s;
                }
              }
            }
          }

          oldDirIdx = 100.0 * fmax(1.0, fabs(TrialState->grad[idxStartIneq - 1]));
        } else {
          mUB = WorkingSet->mConstr;
          if (WorkingSet->mConstr < 1) {
            idxStartIneq = 0;
          } else {
            idxStartIneq = 1;
            if (WorkingSet->mConstr > 1) {
              oldDirIdx = fabs(TrialState->lambdasqp[0]);
              for (idx = 2; idx <= mUB; idx++) {
                s = fabs(TrialState->lambdasqp[idx - 1]);
                if (s > oldDirIdx) {
                  idxStartIneq = idx;
                  oldDirIdx = s;
                }
              }
            }
          }

          oldDirIdx = fabs(TrialState->lambdasqp[idxStartIneq - 1]);
        }

        QPObjective->nvar = WorkingSet->nVar;
        QPObjective->beta = beta;
        QPObjective->rho = oldDirIdx;
        QPObjective->hasLinear = true;
        QPObjective->objtype = 4;
        setProblemType(WorkingSet, 2);
        idxStartIneq = qpoptions->MaxIterations;
        qpoptions->MaxIterations = (qpoptions->MaxIterations + WorkingSet->nVar)
          - nVarOrig;
        memcpy(&dv[0], &TrialState->grad[0], 13U * sizeof(double));
        b_qpoptions = *qpoptions;
        driver(Hessian, dv, TrialState, memspace, WorkingSet, QRManager,
               CholManager, QPObjective, &b_qpoptions, qpoptions->MaxIterations);
        qpoptions->MaxIterations = idxStartIneq;
        if (TrialState->state != -6) {
          MeritFunction->phi = TrialState->sqpFval;
          MeritFunction->linearizedConstrViol = 0.0;
          MeritFunction->penaltyParam = 1.0;
          MeritFunction->phiPrimePlus = fmin((TrialState->fstar - oldDirIdx *
            0.0) - beta / 2.0 * 0.0, 0.0);
          mUB = WorkingSet->isActiveIdx[2];
          idxStartIneq = WorkingSet->nActiveConstr;
          for (idx = mUB; idx <= idxStartIneq; idx++) {
            if (WorkingSet->Wid[idx - 1] == 3) {
              TrialState->lambda[idx - 1] *= (double)memspace->
                workspace_int[WorkingSet->Wlocalidx[idx - 1] - 1];
            }
          }
        }

        QPObjective->nvar = nVarOrig;
        QPObjective->hasLinear = true;
        QPObjective->objtype = 3;
        setProblemType(WorkingSet, 3);
        sortLambdaQP(TrialState->lambda, WorkingSet->nActiveConstr,
                     WorkingSet->sizes, WorkingSet->isActiveIdx, WorkingSet->Wid,
                     WorkingSet->Wlocalidx, memspace->workspace_double);
        if (nVar >= 0) {
          memcpy(&TrialState->delta_x[0], &TrialState->xstar[0], (nVar + 1) *
                 sizeof(double));
        }

        guard1 = true;
      }
      break;

     default:
      idxStartIneq = WorkingSet->nVar - 1;
      if (idxStartIneq >= 0) {
        memcpy(&TrialState->xstarsqp[0], &TrialState->xstarsqp_old[0],
               (idxStartIneq + 1) * sizeof(double));
        memcpy(&TrialState->socDirection[0], &TrialState->xstar[0],
               (idxStartIneq + 1) * sizeof(double));
      }

      memcpy(&TrialState->lambdaStopTest[0], &TrialState->lambda[0], 25U *
             sizeof(double));
      memcpy(&TrialState->xstar[0], &TrialState->xstarsqp[0], (idxStartIneq + 1)
             * sizeof(double));
      memcpy(&dv[0], &TrialState->grad[0], 13U * sizeof(double));
      b_qpoptions = *qpoptions;
      driver(Hessian, dv, TrialState, memspace, WorkingSet, QRManager,
             CholManager, QPObjective, &b_qpoptions, qpoptions->MaxIterations);
      for (idx = 0; idx <= idxStartIneq; idx++) {
        oldDirIdx = TrialState->socDirection[idx];
        TrialState->socDirection[idx] = TrialState->xstar[idx] -
          TrialState->socDirection[idx];
        TrialState->xstar[idx] = oldDirIdx;
      }

      stepSuccess = (b_xnrm2(idxStartIneq + 1, TrialState->socDirection) <= 2.0 *
                     b_xnrm2(idxStartIneq + 1, TrialState->xstar));
      if (!stepSuccess) {
        memcpy(&TrialState->lambda[0], &TrialState->lambdaStopTest[0], 25U *
               sizeof(double));
      } else {
        sortLambdaQP(TrialState->lambda, WorkingSet->nActiveConstr,
                     WorkingSet->sizes, WorkingSet->isActiveIdx, WorkingSet->Wid,
                     WorkingSet->Wlocalidx, memspace->workspace_double);
      }

      checkBoundViolation = stepSuccess;
      if (stepSuccess && (TrialState->state != -6)) {
        for (idx = 0; idx <= nVar; idx++) {
          TrialState->delta_x[idx] = TrialState->xstar[idx] +
            TrialState->socDirection[idx];
        }
      }

      guard1 = true;
      break;
    }

    if (guard1) {
      if (TrialState->state != -6) {
        exitg1 = 1;
      } else {
        oldDirIdx = 0.0;
        s = 1.0;
        for (idx = 0; idx < 12; idx++) {
          oldDirIdx = fmax(oldDirIdx, fabs(TrialState->grad[idx]));
          s = fmax(s, fabs(TrialState->xstar[idx]));
        }

        oldDirIdx = fmax(2.2204460492503131E-16, oldDirIdx / s);
        for (nVarOrig = 0; nVarOrig < 12; nVarOrig++) {
          idxStartIneq = 12 * nVarOrig;
          for (idx = 0; idx < nVarOrig; idx++) {
            Hessian[idxStartIneq + idx] = 0.0;
          }

          Hessian[nVarOrig + 12 * nVarOrig] = oldDirIdx;
          idxStartIneq += nVarOrig;
          mUB = 10 - nVarOrig;
          if (mUB >= 0) {
            memset(&Hessian[idxStartIneq + 1], 0, (((mUB + idxStartIneq) -
                     idxStartIneq) + 1) * sizeof(double));
          }
        }
      }
    }
  } while (exitg1 == 0);

  if (checkBoundViolation) {
    idxStartIneq = WorkingSet->sizes[3];
    mUB = WorkingSet->sizes[4];
    memcpy(&dv[0], &TrialState->delta_x[0], 13U * sizeof(double));
    for (idx = 0; idx < idxStartIneq; idx++) {
      oldDirIdx = dv[WorkingSet->indexLB[idx] - 1];
      s = (TrialState->xstarsqp[WorkingSet->indexLB[idx] - 1] + oldDirIdx) -
        lb[WorkingSet->indexLB[idx] - 1];
      if (s < 0.0) {
        dv[WorkingSet->indexLB[idx] - 1] = oldDirIdx - s;
        TrialState->xstar[WorkingSet->indexLB[idx] - 1] -= s;
      }
    }

    for (idx = 0; idx < mUB; idx++) {
      oldDirIdx = dv[WorkingSet->indexUB[idx] - 1];
      s = (ub[WorkingSet->indexUB[idx] - 1] - TrialState->xstarsqp
           [WorkingSet->indexUB[idx] - 1]) - oldDirIdx;
      if (s < 0.0) {
        dv[WorkingSet->indexUB[idx] - 1] = oldDirIdx + s;
        TrialState->xstar[WorkingSet->indexUB[idx] - 1] += s;
      }
    }

    memcpy(&TrialState->delta_x[0], &dv[0], 13U * sizeof(double));
  }

  return stepSuccess;
}

/*
 * Arguments    : const double A[72]
 *                double U[72]
 *                double s[6]
 *                double V[36]
 * Return Type  : void
 */
static void svd(const double A[72], double U[72], double s[6], double V[36])
{
  double b_A[72];
  double Vf[36];
  double work[12];
  double b_s[6];
  double e[6];
  double nrm;
  double rt;
  double sm;
  double snorm;
  double sqds;
  int i;
  int ii;
  int jj;
  int k;
  int m;
  int q;
  int qp1;
  int qp1jj;
  int qq;
  memcpy(&b_A[0], &A[0], 72U * sizeof(double));
  for (i = 0; i < 6; i++) {
    b_s[i] = 0.0;
    e[i] = 0.0;
  }

  memset(&work[0], 0, 12U * sizeof(double));
  memset(&U[0], 0, 72U * sizeof(double));
  memset(&Vf[0], 0, 36U * sizeof(double));
  for (q = 0; q < 6; q++) {
    bool apply_transform;
    qp1 = q + 2;
    qq = (q + 12 * q) + 1;
    apply_transform = false;
    nrm = d_xnrm2(12 - q, b_A, qq);
    if (nrm > 0.0) {
      apply_transform = true;
      if (b_A[qq - 1] < 0.0) {
        nrm = -nrm;
      }

      b_s[q] = nrm;
      if (fabs(nrm) >= 1.0020841800044864E-292) {
        nrm = 1.0 / nrm;
        qp1jj = (qq - q) + 11;
        for (k = qq; k <= qp1jj; k++) {
          b_A[k - 1] *= nrm;
        }
      } else {
        qp1jj = (qq - q) + 11;
        for (k = qq; k <= qp1jj; k++) {
          b_A[k - 1] /= b_s[q];
        }
      }

      b_A[qq - 1]++;
      b_s[q] = -b_s[q];
    } else {
      b_s[q] = 0.0;
    }

    for (jj = qp1; jj < 7; jj++) {
      i = q + 12 * (jj - 1);
      if (apply_transform) {
        xaxpy(12 - q, -(xdotc(12 - q, b_A, qq, b_A, i + 1) / b_A[q + 12 * q]),
              qq, b_A, i + 1);
      }

      e[jj - 1] = b_A[i];
    }

    for (ii = q + 1; ii < 13; ii++) {
      i = (ii + 12 * q) - 1;
      U[i] = b_A[i];
    }

    if (q + 1 <= 4) {
      nrm = e_xnrm2(5 - q, e, q + 2);
      if (nrm == 0.0) {
        e[q] = 0.0;
      } else {
        if (e[q + 1] < 0.0) {
          e[q] = -nrm;
        } else {
          e[q] = nrm;
        }

        nrm = e[q];
        if (fabs(e[q]) >= 1.0020841800044864E-292) {
          nrm = 1.0 / e[q];
          for (k = qp1; k < 7; k++) {
            e[k - 1] *= nrm;
          }
        } else {
          for (k = qp1; k < 7; k++) {
            e[k - 1] /= nrm;
          }
        }

        e[q + 1]++;
        e[q] = -e[q];
        for (ii = qp1; ii < 13; ii++) {
          work[ii - 1] = 0.0;
        }

        for (jj = qp1; jj < 7; jj++) {
          b_xaxpy(11 - q, e[jj - 1], b_A, (q + 12 * (jj - 1)) + 2, work, q + 2);
        }

        for (jj = qp1; jj < 7; jj++) {
          c_xaxpy(11 - q, -e[jj - 1] / e[q + 1], work, q + 2, b_A, (q + 12 * (jj
                    - 1)) + 2);
        }
      }

      for (ii = qp1; ii < 7; ii++) {
        Vf[(ii + 6 * q) - 1] = e[ii - 1];
      }
    }
  }

  m = 4;
  e[4] = b_A[64];
  e[5] = 0.0;
  for (q = 5; q >= 0; q--) {
    qp1 = q + 2;
    qq = q + 12 * q;
    if (b_s[q] != 0.0) {
      for (jj = qp1; jj < 7; jj++) {
        i = (q + 12 * (jj - 1)) + 1;
        xaxpy(12 - q, -(xdotc(12 - q, U, qq + 1, U, i) / U[qq]), qq + 1, U, i);
      }

      for (ii = q + 1; ii < 13; ii++) {
        i = (ii + 12 * q) - 1;
        U[i] = -U[i];
      }

      U[qq]++;
      for (ii = 0; ii < q; ii++) {
        U[ii + 12 * q] = 0.0;
      }
    } else {
      memset(&U[q * 12], 0, 12U * sizeof(double));
      U[qq] = 1.0;
    }
  }

  for (q = 5; q >= 0; q--) {
    if ((q + 1 <= 4) && (e[q] != 0.0)) {
      qp1 = q + 2;
      i = (q + 6 * q) + 2;
      for (jj = qp1; jj < 7; jj++) {
        qp1jj = (q + 6 * (jj - 1)) + 2;
        d_xaxpy(5 - q, -(b_xdotc(5 - q, Vf, i, Vf, qp1jj) / Vf[i - 1]), i, Vf,
                qp1jj);
      }
    }

    for (ii = 0; ii < 6; ii++) {
      Vf[ii + 6 * q] = 0.0;
    }

    Vf[q + 6 * q] = 1.0;
  }

  qq = 0;
  snorm = 0.0;
  for (q = 0; q < 6; q++) {
    nrm = b_s[q];
    if (nrm != 0.0) {
      rt = fabs(nrm);
      nrm /= rt;
      b_s[q] = rt;
      if (q + 1 < 6) {
        e[q] /= nrm;
      }

      i = 12 * q;
      qp1jj = i + 12;
      for (k = i + 1; k <= qp1jj; k++) {
        U[k - 1] *= nrm;
      }
    }

    if (q + 1 < 6) {
      nrm = e[q];
      if (nrm != 0.0) {
        rt = fabs(nrm);
        nrm = rt / nrm;
        e[q] = rt;
        b_s[q + 1] *= nrm;
        i = 6 * (q + 1);
        qp1jj = i + 6;
        for (k = i + 1; k <= qp1jj; k++) {
          Vf[k - 1] *= nrm;
        }
      }
    }

    snorm = fmax(snorm, fmax(fabs(b_s[q]), fabs(e[q])));
  }

  while ((m + 2 > 0) && (qq < 75)) {
    bool exitg1;
    jj = m + 1;
    ii = m + 1;
    exitg1 = false;
    while (!(exitg1 || (ii == 0))) {
      nrm = fabs(e[ii - 1]);
      if ((nrm <= 2.2204460492503131E-16 * (fabs(b_s[ii - 1]) + fabs(b_s[ii]))) ||
          (nrm <= 1.0020841800044864E-292) || ((qq > 20) && (nrm <=
            2.2204460492503131E-16 * snorm))) {
        e[ii - 1] = 0.0;
        exitg1 = true;
      } else {
        ii--;
      }
    }

    if (ii == m + 1) {
      i = 4;
    } else {
      qp1jj = m + 2;
      i = m + 2;
      exitg1 = false;
      while ((!exitg1) && (i >= ii)) {
        qp1jj = i;
        if (i == ii) {
          exitg1 = true;
        } else {
          nrm = 0.0;
          if (i < m + 2) {
            nrm = fabs(e[i - 1]);
          }

          if (i > ii + 1) {
            nrm += fabs(e[i - 2]);
          }

          rt = fabs(b_s[i - 1]);
          if ((rt <= 2.2204460492503131E-16 * nrm) || (rt <=
               1.0020841800044864E-292)) {
            b_s[i - 1] = 0.0;
            exitg1 = true;
          } else {
            i--;
          }
        }
      }

      if (qp1jj == ii) {
        i = 3;
      } else if (qp1jj == m + 2) {
        i = 1;
      } else {
        i = 2;
        ii = qp1jj;
      }
    }

    switch (i) {
     case 1:
      {
        rt = e[m];
        e[m] = 0.0;
        for (k = jj; k >= ii + 1; k--) {
          xrotg(&b_s[k - 1], &rt, &sm, &sqds);
          if (k > ii + 1) {
            double b;
            b = e[k - 2];
            rt = -sqds * b;
            e[k - 2] = b * sm;
          }

          xrot(Vf, 6 * (k - 1) + 1, 6 * (m + 1) + 1, sm, sqds);
        }
      }
      break;

     case 2:
      {
        rt = e[ii - 1];
        e[ii - 1] = 0.0;
        for (k = ii + 1; k <= m + 2; k++) {
          double b;
          xrotg(&b_s[k - 1], &rt, &sm, &sqds);
          b = e[k - 1];
          rt = -sqds * b;
          e[k - 1] = b * sm;
          b_xrot(U, 12 * (k - 1) + 1, 12 * (ii - 1) + 1, sm, sqds);
        }
      }
      break;

     case 3:
      {
        double b;
        double scale;
        nrm = b_s[m + 1];
        scale = fmax(fmax(fmax(fmax(fabs(nrm), fabs(b_s[m])), fabs(e[m])), fabs
                          (b_s[ii])), fabs(e[ii]));
        sm = nrm / scale;
        nrm = b_s[m] / scale;
        rt = e[m] / scale;
        sqds = b_s[ii] / scale;
        b = ((nrm + sm) * (nrm - sm) + rt * rt) / 2.0;
        nrm = sm * rt;
        nrm *= nrm;
        if ((b != 0.0) || (nrm != 0.0)) {
          rt = sqrt(b * b + nrm);
          if (b < 0.0) {
            rt = -rt;
          }

          rt = nrm / (b + rt);
        } else {
          rt = 0.0;
        }

        rt += (sqds + sm) * (sqds - sm);
        nrm = sqds * (e[ii] / scale);
        for (k = ii + 1; k <= jj; k++) {
          xrotg(&rt, &nrm, &sm, &sqds);
          if (k > ii + 1) {
            e[k - 2] = rt;
          }

          nrm = e[k - 1];
          b = b_s[k - 1];
          e[k - 1] = sm * nrm - sqds * b;
          rt = sqds * b_s[k];
          b_s[k] *= sm;
          xrot(Vf, 6 * (k - 1) + 1, 6 * k + 1, sm, sqds);
          b_s[k - 1] = sm * b + sqds * nrm;
          xrotg(&b_s[k - 1], &rt, &sm, &sqds);
          rt = sm * e[k - 1] + sqds * b_s[k];
          b_s[k] = -sqds * e[k - 1] + sm * b_s[k];
          nrm = sqds * e[k];
          e[k] *= sm;
          b_xrot(U, 12 * (k - 1) + 1, 12 * k + 1, sm, sqds);
        }

        e[m] = rt;
        qq++;
      }
      break;

     default:
      if (b_s[ii] < 0.0) {
        b_s[ii] = -b_s[ii];
        i = 6 * ii;
        qp1jj = i + 6;
        for (k = i + 1; k <= qp1jj; k++) {
          Vf[k - 1] = -Vf[k - 1];
        }
      }

      qp1 = ii + 1;
      while ((ii + 1 < 6) && (b_s[ii] < b_s[qp1])) {
        rt = b_s[ii];
        b_s[ii] = b_s[qp1];
        b_s[qp1] = rt;
        xswap(Vf, 6 * ii + 1, 6 * (ii + 1) + 1);
        b_xswap(U, 12 * ii + 1, 12 * (ii + 1) + 1);
        ii = qp1;
        qp1++;
      }

      qq = 0;
      m--;
      break;
    }
  }

  for (k = 0; k < 6; k++) {
    s[k] = b_s[k];
    for (i = 0; i < 6; i++) {
      qp1jj = i + 6 * k;
      V[qp1jj] = Vf[qp1jj];
    }
  }
}

/*
 * Arguments    : b_struct_T *MeritFunction
 *                const k_struct_T *WorkingSet
 *                i_struct_T *TrialState
 *                const double lb[12]
 *                const double ub[12]
 *                bool *Flags_gradOK
 *                bool *Flags_fevalOK
 *                bool *Flags_done
 *                bool *Flags_stepAccepted
 *                bool *Flags_failedLineSearch
 *                int *Flags_stepType
 * Return Type  : void
 */
static void test_exit(b_struct_T *MeritFunction, const k_struct_T *WorkingSet,
                      i_struct_T *TrialState, const double lb[12], const double
                      ub[12], bool *Flags_gradOK, bool *Flags_fevalOK, bool
                      *Flags_done, bool *Flags_stepAccepted, bool
                      *Flags_failedLineSearch, int *Flags_stepType)
{
  double s;
  double smax;
  int b_k;
  int idx_max;
  int k;
  int mLB;
  int mLambda;
  int mUB;
  int nVar;
  bool exitg1;
  bool isFeasible;
  *Flags_fevalOK = true;
  *Flags_done = false;
  *Flags_stepAccepted = false;
  *Flags_failedLineSearch = false;
  *Flags_stepType = 1;
  nVar = WorkingSet->nVar;
  mLB = WorkingSet->sizes[3];
  mUB = WorkingSet->sizes[4];
  mLambda = ((WorkingSet->sizes[0] + WorkingSet->sizes[3]) + WorkingSet->sizes[4])
    - 1;
  if (mLambda + 1 < 400) {
    if (mLambda >= 0) {
      memcpy(&TrialState->lambdaStopTest[0], &TrialState->lambdasqp[0], (mLambda
              + 1) * sizeof(double));
    }
  } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

    for (k = 0; k <= mLambda; k++) {
      TrialState->lambdaStopTest[k] = TrialState->lambdasqp[k];
    }
  }

  computeGradLag(TrialState->gradLag, WorkingSet->nVar, TrialState->grad,
                 WorkingSet->indexFixed, WorkingSet->sizes[0],
                 WorkingSet->indexLB, WorkingSet->sizes[3], WorkingSet->indexUB,
                 WorkingSet->sizes[4], TrialState->lambdaStopTest);
  if (WorkingSet->nVar < 1) {
    idx_max = 0;
  } else {
    idx_max = 1;
    if (WorkingSet->nVar > 1) {
      smax = fabs(TrialState->grad[0]);
      for (b_k = 2; b_k <= nVar; b_k++) {
        s = fabs(TrialState->grad[b_k - 1]);
        if (s > smax) {
          idx_max = b_k;
          smax = s;
        }
      }
    }
  }

  s = fmax(1.0, fabs(TrialState->grad[idx_max - 1]));
  if (rtIsInf(s)) {
    s = 1.0;
  }

  smax = 0.0;
  for (nVar = 0; nVar < mLB; nVar++) {
    idx_max = WorkingSet->indexLB[nVar] - 1;
    smax = fmax(smax, lb[idx_max] - TrialState->xstarsqp[idx_max]);
  }

  for (nVar = 0; nVar < mUB; nVar++) {
    idx_max = WorkingSet->indexUB[nVar] - 1;
    smax = fmax(smax, TrialState->xstarsqp[idx_max] - ub[idx_max]);
  }

  MeritFunction->nlpPrimalFeasError = smax;
  MeritFunction->feasRelativeFactor = fmax(1.0, smax);
  isFeasible = (smax <= 1.0E-6 * MeritFunction->feasRelativeFactor);
  *Flags_gradOK = true;
  smax = 0.0;
  nVar = 0;
  exitg1 = false;
  while ((!exitg1) && (nVar <= WorkingSet->nVar - 1)) {
    *Flags_gradOK = ((!rtIsInf(TrialState->gradLag[nVar])) && (!rtIsNaN
      (TrialState->gradLag[nVar])));
    if (!*Flags_gradOK) {
      exitg1 = true;
    } else {
      smax = fmax(smax, fabs(TrialState->gradLag[nVar]));
      nVar++;
    }
  }

  MeritFunction->nlpDualFeasError = smax;
  if (!*Flags_gradOK) {
    *Flags_done = true;
    if (isFeasible) {
      TrialState->sqpExitFlag = 2;
    } else {
      TrialState->sqpExitFlag = -2;
    }
  } else {
    MeritFunction->nlpComplError = 0.0;
    MeritFunction->firstOrderOpt = smax;
    if (mLambda + 1 < 400) {
      if (mLambda >= 0) {
        memcpy(&TrialState->lambdaStopTestPrev[0], &TrialState->lambdaStopTest[0],
               (mLambda + 1) * sizeof(double));
      }
    } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

      for (k = 0; k <= mLambda; k++) {
        TrialState->lambdaStopTestPrev[k] = TrialState->lambdaStopTest[k];
      }
    }

    if (isFeasible && (smax <= 1.0E-9 * s)) {
      *Flags_done = true;
      TrialState->sqpExitFlag = 1;
    } else if (isFeasible && (TrialState->sqpFval < -1.0E+20)) {
      *Flags_done = true;
      TrialState->sqpExitFlag = -3;
    }
  }
}

/*
 * Arguments    : void
 * Return Type  : void
 */
static void tic(void)
{
  coderTimespec b_timespec;
  if (!freq_not_empty) {
    freq_not_empty = true;
    coderInitTimeFunctions(&freq);
  }

  coderTimeClockGettimeMonotonic(&b_timespec, freq);
  timeKeeper(b_timespec.tv_sec, b_timespec.tv_nsec);
}

/*
 * Arguments    : double newTime_tv_sec
 *                double newTime_tv_nsec
 * Return Type  : void
 */
static void timeKeeper(double newTime_tv_sec, double newTime_tv_nsec)
{
  if (!savedTime_not_empty) {
    if (!freq_not_empty) {
      freq_not_empty = true;
      coderInitTimeFunctions(&freq);
    }

    coderTimeClockGettimeMonotonic(&savedTime, freq);
    savedTime_not_empty = true;
  }

  savedTime.tv_sec = newTime_tv_sec;
  savedTime.tv_nsec = newTime_tv_nsec;
}

/*
 * Arguments    : void
 * Return Type  : double
 */
static double toc(void)
{
  coderTimespec b_timespec;
  double tstart_tv_nsec;
  double tstart_tv_sec;
  b_timeKeeper(&tstart_tv_sec, &tstart_tv_nsec);
  if (!freq_not_empty) {
    freq_not_empty = true;
    coderInitTimeFunctions(&freq);
  }

  coderTimeClockGettimeMonotonic(&b_timespec, freq);
  return (b_timespec.tv_sec - tstart_tv_sec) + (b_timespec.tv_nsec -
    tstart_tv_nsec) / 1.0E+9;
}

/*
 * Arguments    : int n
 *                double a
 *                int ix0
 *                double y[72]
 *                int iy0
 * Return Type  : void
 */
static void xaxpy(int n, double a, int ix0, double y[72], int iy0)
{
  int k;
  if (!(a == 0.0)) {
    int i;
    i = n - 1;
    for (k = 0; k <= i; k++) {
      int i1;
      i1 = (iy0 + k) - 1;
      y[i1] += a * y[(ix0 + k) - 1];
    }
  }
}

/*
 * Arguments    : int n
 *                const double x[72]
 *                int ix0
 *                const double y[72]
 *                int iy0
 * Return Type  : double
 */
static double xdotc(int n, const double x[72], int ix0, const double y[72], int
                    iy0)
{
  double d;
  int k;
  d = 0.0;
  for (k = 0; k < n; k++) {
    d += x[(ix0 + k) - 1] * y[(iy0 + k) - 1];
  }

  return d;
}

/*
 * Arguments    : int m
 *                int n
 *                int k
 *                const double A[144]
 *                int lda
 *                const double B[625]
 *                int ib0
 *                double C[325]
 * Return Type  : void
 */
static void xgemm(int m, int n, int k, const double A[144], int lda, const
                  double B[625], int ib0, double C[325])
{
  int cr;
  int ib;
  int ic;
  if ((m != 0) && (n != 0)) {
    int br;
    int i;
    int i1;
    int lastColC;
    br = ib0;
    lastColC = 25 * (n - 1);
    for (cr = 0; cr <= lastColC; cr += 25) {
      i = cr + 1;
      i1 = cr + m;
      if (i <= i1) {
        memset(&C[i + -1], 0, ((i1 - i) + 1) * sizeof(double));
      }
    }

    for (cr = 0; cr <= lastColC; cr += 25) {
      int ar;
      ar = -1;
      i = br + k;
      for (ib = br; ib < i; ib++) {
        int i2;
        i1 = cr + 1;
        i2 = cr + m;
        for (ic = i1; ic <= i2; ic++) {
          C[ic - 1] += B[ib - 1] * A[(ar + ic) - cr];
        }

        ar += lda;
      }

      br += 25;
    }
  }
}

/*
 * Arguments    : int m
 *                int n
 *                const double A[625]
 *                const double x[13]
 *                double y[325]
 * Return Type  : void
 */
static void xgemv(int m, int n, const double A[625], const double x[13], double
                  y[325])
{
  int ia;
  int iac;
  int iy;
  if (m != 0) {
    int i;
    if (n < 400) {
      if (n - 1 >= 0) {
        memset(&y[0], 0, n * sizeof(double));
      }
    } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

      for (iy = 0; iy < n; iy++) {
        y[iy] = 0.0;
      }
    }

    i = 25 * (n - 1) + 1;
    for (iac = 1; iac <= i; iac += 25) {
      double c;
      int i1;
      c = 0.0;
      i1 = (iac + m) - 1;
      for (ia = iac; ia <= i1; ia++) {
        c += A[ia - 1] * x[ia - iac];
      }

      i1 = div_nde_s32_floor(iac - 1, 25);
      y[i1] += c;
    }
  }
}

/*
 * Arguments    : double A[625]
 *                int m
 *                int n
 *                int jpvt[25]
 *                double tau[25]
 * Return Type  : void
 */
static void xgeqp3(double A[625], int m, int n, int jpvt[25], double tau[25])
{
  double vn1[25];
  double vn2[25];
  double work[25];
  double d;
  double temp;
  int b_i;
  int j;
  int k;
  int minmn;
  int pvt;
  if (m <= n) {
    minmn = m;
  } else {
    minmn = n;
  }

  memset(&tau[0], 0, 25U * sizeof(double));
  if (minmn < 1) {
    if (n < 400) {
      for (j = 0; j < n; j++) {
        jpvt[j] = j + 1;
      }
    } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

      for (j = 0; j < n; j++) {
        jpvt[j] = j + 1;
      }
    }
  } else {
    int i;
    int ix;
    int iy;
    int nfxd;
    int temp_tmp;
    nfxd = 0;
    for (pvt = 0; pvt < n; pvt++) {
      if (jpvt[pvt] != 0) {
        nfxd++;
        if (pvt + 1 != nfxd) {
          ix = pvt * 25;
          iy = (nfxd - 1) * 25;
          for (k = 0; k < m; k++) {
            temp_tmp = ix + k;
            temp = A[temp_tmp];
            i = iy + k;
            A[temp_tmp] = A[i];
            A[i] = temp;
          }

          jpvt[pvt] = jpvt[nfxd - 1];
          jpvt[nfxd - 1] = pvt + 1;
        } else {
          jpvt[pvt] = pvt + 1;
        }
      } else {
        jpvt[pvt] = pvt + 1;
      }
    }

    if (nfxd > minmn) {
      nfxd = minmn;
    }

    qrf(A, m, n, nfxd, tau);
    if (nfxd < minmn) {
      memset(&work[0], 0, 25U * sizeof(double));
      memset(&vn1[0], 0, 25U * sizeof(double));
      memset(&vn2[0], 0, 25U * sizeof(double));
      i = nfxd + 1;
      iy = nfxd + 1;
      if (n - nfxd < 400) {
        for (j = i; j <= n; j++) {
          d = xnrm2(m - nfxd, A, (nfxd + (j - 1) * 25) + 1);
          vn1[j - 1] = d;
          vn2[j - 1] = d;
        }
      } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4) \
 private(d)

        for (j = iy; j <= n; j++) {
          d = xnrm2(m - nfxd, A, (nfxd + (j - 1) * 25) + 1);
          vn1[j - 1] = d;
          vn2[j - 1] = d;
        }
      }

      i = nfxd + 1;
      for (b_i = i; b_i <= minmn; b_i++) {
        double d1;
        double s;
        int ii;
        int ip1;
        int mmi;
        int nmi;
        ip1 = b_i + 1;
        nfxd = (b_i - 1) * 25;
        ii = (nfxd + b_i) - 1;
        nmi = (n - b_i) + 1;
        mmi = m - b_i;
        if (nmi < 1) {
          iy = -2;
        } else {
          iy = -1;
          if (nmi > 1) {
            temp = fabs(vn1[b_i - 1]);
            for (k = 2; k <= nmi; k++) {
              s = fabs(vn1[(b_i + k) - 2]);
              if (s > temp) {
                iy = k - 2;
                temp = s;
              }
            }
          }
        }

        pvt = b_i + iy;
        if (pvt + 1 != b_i) {
          ix = pvt * 25;
          for (k = 0; k < m; k++) {
            temp_tmp = ix + k;
            temp = A[temp_tmp];
            iy = nfxd + k;
            A[temp_tmp] = A[iy];
            A[iy] = temp;
          }

          iy = jpvt[pvt];
          jpvt[pvt] = jpvt[b_i - 1];
          jpvt[b_i - 1] = iy;
          vn1[pvt] = vn1[b_i - 1];
          vn2[pvt] = vn2[b_i - 1];
        }

        if (b_i < m) {
          temp = A[ii];
          d1 = xzlarfg(mmi + 1, &temp, A, ii + 2);
          tau[b_i - 1] = d1;
          A[ii] = temp;
        } else {
          d1 = 0.0;
          tau[b_i - 1] = 0.0;
        }

        if (b_i < n) {
          temp = A[ii];
          A[ii] = 1.0;
          xzlarf(mmi + 1, nmi - 1, ii + 1, d1, A, ii + 26, work);
          A[ii] = temp;
        }

        for (pvt = ip1; pvt <= n; pvt++) {
          iy = b_i + (pvt - 1) * 25;
          d1 = vn1[pvt - 1];
          if (d1 != 0.0) {
            temp = fabs(A[iy - 1]) / d1;
            temp = 1.0 - temp * temp;
            if (temp < 0.0) {
              temp = 0.0;
            }

            s = d1 / vn2[pvt - 1];
            s = temp * (s * s);
            if (s <= 1.4901161193847656E-8) {
              if (b_i < m) {
                d1 = xnrm2(mmi, A, iy + 1);
                vn1[pvt - 1] = d1;
                vn2[pvt - 1] = d1;
              } else {
                vn1[pvt - 1] = 0.0;
                vn2[pvt - 1] = 0.0;
              }
            } else {
              vn1[pvt - 1] = d1 * sqrt(temp);
            }
          }
        }
      }
    }
  }
}

/*
 * Arguments    : int n
 *                const double x[625]
 *                int ix0
 * Return Type  : double
 */
static double xnrm2(int n, const double x[625], int ix0)
{
  double y;
  int k;
  y = 0.0;
  if (n >= 1) {
    if (n == 1) {
      y = fabs(x[ix0 - 1]);
    } else {
      double scale;
      int kend;
      scale = 3.3121686421112381E-170;
      kend = (ix0 + n) - 1;
      for (k = ix0; k <= kend; k++) {
        double absxk;
        absxk = fabs(x[k - 1]);
        if (absxk > scale) {
          double t;
          t = scale / absxk;
          y = y * t * t + 1.0;
          scale = absxk;
        } else {
          double t;
          t = absxk / scale;
          y += t * t;
        }
      }

      y = scale * sqrt(y);
    }
  }

  return y;
}

/*
 * Arguments    : int n
 *                double A[625]
 * Return Type  : int
 */
static int xpotrf(int n, double A[625])
{
  int ia;
  int iac;
  int info;
  int j;
  int nmj;
  bool exitg1;
  info = 0;
  j = 0;
  exitg1 = false;
  while ((!exitg1) && (j <= n - 1)) {
    double c;
    double ssq;
    int idxA1j;
    int idxAjj;
    idxA1j = j * 25;
    idxAjj = idxA1j + j;
    ssq = 0.0;
    if (j >= 1) {
      for (nmj = 0; nmj < j; nmj++) {
        c = A[idxA1j + nmj];
        ssq += c * c;
      }
    }

    ssq = A[idxAjj] - ssq;
    if (ssq > 0.0) {
      ssq = sqrt(ssq);
      A[idxAjj] = ssq;
      if (j + 1 < n) {
        int i;
        int ia0;
        int idxAjjp1;
        nmj = (n - j) - 2;
        ia0 = idxA1j + 26;
        idxAjjp1 = idxAjj + 26;
        if ((j != 0) && (nmj + 1 != 0)) {
          i = (idxA1j + 25 * nmj) + 26;
          for (iac = ia0; iac <= i; iac += 25) {
            int i1;
            c = 0.0;
            i1 = (iac + j) - 1;
            for (ia = iac; ia <= i1; ia++) {
              c += A[ia - 1] * A[(idxA1j + ia) - iac];
            }

            i1 = (idxAjj + div_nde_s32_floor((iac - idxA1j) - 26, 25) * 25) + 25;
            A[i1] += -c;
          }
        }

        ssq = 1.0 / ssq;
        i = (idxAjj + 25 * nmj) + 26;
        for (nmj = idxAjjp1; nmj <= i; nmj += 25) {
          A[nmj - 1] *= ssq;
        }
      }

      j++;
    } else {
      A[idxAjj] = ssq;
      info = j + 1;
      exitg1 = true;
    }
  }

  return info;
}

/*
 * Arguments    : double x[36]
 *                int ix0
 *                int iy0
 *                double c
 *                double s
 * Return Type  : void
 */
static void xrot(double x[36], int ix0, int iy0, double c, double s)
{
  int k;
  for (k = 0; k < 6; k++) {
    double b_temp_tmp;
    double d_temp_tmp;
    int c_temp_tmp;
    int temp_tmp;
    temp_tmp = (iy0 + k) - 1;
    b_temp_tmp = x[temp_tmp];
    c_temp_tmp = (ix0 + k) - 1;
    d_temp_tmp = x[c_temp_tmp];
    x[temp_tmp] = c * b_temp_tmp - s * d_temp_tmp;
    x[c_temp_tmp] = c * d_temp_tmp + s * b_temp_tmp;
  }
}

/*
 * Arguments    : double *a
 *                double *b
 *                double *c
 *                double *s
 * Return Type  : void
 */
static void xrotg(double *a, double *b, double *c, double *s)
{
  double absa;
  double absb;
  double roe;
  double scale;
  roe = *b;
  absa = fabs(*a);
  absb = fabs(*b);
  if (absa > absb) {
    roe = *a;
  }

  scale = absa + absb;
  if (scale == 0.0) {
    *s = 0.0;
    *c = 1.0;
    *a = 0.0;
    *b = 0.0;
  } else {
    double ads;
    double bds;
    ads = absa / scale;
    bds = absb / scale;
    scale *= sqrt(ads * ads + bds * bds);
    if (roe < 0.0) {
      scale = -scale;
    }

    *c = *a / scale;
    *s = *b / scale;
    if (absa > absb) {
      *b = *s;
    } else if (*c != 0.0) {
      *b = 1.0 / *c;
    } else {
      *b = 1.0;
    }

    *a = scale;
  }
}

/*
 * Arguments    : double x[36]
 *                int ix0
 *                int iy0
 * Return Type  : void
 */
static void xswap(double x[36], int ix0, int iy0)
{
  int k;
  for (k = 0; k < 6; k++) {
    double temp;
    int i;
    int temp_tmp;
    temp_tmp = (ix0 + k) - 1;
    temp = x[temp_tmp];
    i = (iy0 + k) - 1;
    x[temp_tmp] = x[i];
    x[i] = temp;
  }
}

/*
 * Arguments    : int m
 *                int n
 *                int iv0
 *                double tau
 *                double C[625]
 *                int ic0
 *                double work[25]
 * Return Type  : void
 */
static void xzlarf(int m, int n, int iv0, double tau, double C[625], int ic0,
                   double work[25])
{
  int i;
  int ia;
  int iac;
  int iy;
  int lastc;
  int lastv;
  if (tau != 0.0) {
    bool exitg2;
    lastv = m;
    i = iv0 + m;
    while ((lastv > 0) && (C[i - 2] == 0.0)) {
      lastv--;
      i--;
    }

    lastc = n - 1;
    exitg2 = false;
    while ((!exitg2) && (lastc + 1 > 0)) {
      int exitg1;
      i = ic0 + lastc * 25;
      ia = i;
      do {
        exitg1 = 0;
        if (ia <= (i + lastv) - 1) {
          if (C[ia - 1] != 0.0) {
            exitg1 = 1;
          } else {
            ia++;
          }
        } else {
          lastc--;
          exitg1 = 2;
        }
      } while (exitg1 == 0);

      if (exitg1 == 1) {
        exitg2 = true;
      }
    }
  } else {
    lastv = 0;
    lastc = -1;
  }

  if (lastv > 0) {
    double c;
    int b_i;
    if (lastc + 1 != 0) {
      if (lastc + 1 < 400) {
        if (lastc >= 0) {
          memset(&work[0], 0, (lastc + 1) * sizeof(double));
        }
      } else {

#pragma omp parallel for \
 num_threads(4 > omp_get_max_threads() ? omp_get_max_threads() : 4)

        for (iy = 0; iy <= lastc; iy++) {
          work[iy] = 0.0;
        }
      }

      b_i = ic0 + 25 * lastc;
      for (iac = ic0; iac <= b_i; iac += 25) {
        c = 0.0;
        i = (iac + lastv) - 1;
        for (ia = iac; ia <= i; ia++) {
          c += C[ia - 1] * C[((iv0 + ia) - iac) - 1];
        }

        i = div_nde_s32_floor(iac - ic0, 25);
        work[i] += c;
      }
    }

    if (!(-tau == 0.0)) {
      i = ic0;
      for (iac = 0; iac <= lastc; iac++) {
        if (work[iac] != 0.0) {
          c = work[iac] * -tau;
          b_i = lastv + i;
          for (ia = i; ia < b_i; ia++) {
            C[ia - 1] += C[((iv0 + ia) - i) - 1] * c;
          }
        }

        i += 25;
      }
    }
  }
}

/*
 * Arguments    : int n
 *                double *alpha1
 *                double x[625]
 *                int ix0
 * Return Type  : double
 */
static double xzlarfg(int n, double *alpha1, double x[625], int ix0)
{
  double tau;
  int k;
  tau = 0.0;
  if (n > 0) {
    double xnorm;
    xnorm = xnrm2(n - 1, x, ix0);
    if (xnorm != 0.0) {
      double beta1;
      beta1 = rt_hypotd_snf(*alpha1, xnorm);
      if (*alpha1 >= 0.0) {
        beta1 = -beta1;
      }

      if (fabs(beta1) < 1.0020841800044864E-292) {
        int i;
        int knt;
        knt = 0;
        i = (ix0 + n) - 2;
        do {
          knt++;
          for (k = ix0; k <= i; k++) {
            x[k - 1] *= 9.9792015476736E+291;
          }

          beta1 *= 9.9792015476736E+291;
          *alpha1 *= 9.9792015476736E+291;
        } while ((fabs(beta1) < 1.0020841800044864E-292) && (knt < 20));

        beta1 = rt_hypotd_snf(*alpha1, xnrm2(n - 1, x, ix0));
        if (*alpha1 >= 0.0) {
          beta1 = -beta1;
        }

        tau = (beta1 - *alpha1) / beta1;
        xnorm = 1.0 / (*alpha1 - beta1);
        for (k = ix0; k <= i; k++) {
          x[k - 1] *= xnorm;
        }

        for (k = 0; k < knt; k++) {
          beta1 *= 1.0020841800044864E-292;
        }

        *alpha1 = beta1;
      } else {
        int i;
        tau = (beta1 - *alpha1) / beta1;
        xnorm = 1.0 / (*alpha1 - beta1);
        i = (ix0 + n) - 2;
        for (k = ix0; k <= i; k++) {
          x[k - 1] *= xnorm;
        }

        *alpha1 = beta1;
      }
    }
  }

  return tau;
}

/*
 * %% Testing variables:
 *
 * Arguments    : double K_p_T
 *                double K_p_M
 *                double m
 *                double I_xx
 *                double I_yy
 *                double I_zz
 *                double l_1
 *                double l_2
 *                double l_3
 *                double l_4
 *                double l_z
 *                double Phi
 *                double Theta
 *                double Psi
 *                double Omega_1
 *                double Omega_2
 *                double Omega_3
 *                double Omega_4
 *                double b_1
 *                double b_2
 *                double b_3
 *                double b_4
 *                double g_1
 *                double g_2
 *                double g_3
 *                double g_4
 *                double W_act_motor_const
 *                double W_act_motor_speed
 *                double W_act_tilt_el_const
 *                double W_act_tilt_el_speed
 *                double W_act_tilt_az_const
 *                double W_act_tilt_az_speed
 *                double W_act_theta_const
 *                double W_act_theta_speed
 *                double W_act_phi_const
 *                double W_act_phi_speed
 *                double W_dv_1
 *                double W_dv_2
 *                double W_dv_3
 *                double W_dv_4
 *                double W_dv_5
 *                double W_dv_6
 *                double max_omega
 *                double min_omega
 *                double max_b
 *                double min_b
 *                double max_g
 *                double min_g
 *                double max_theta
 *                double min_theta
 *                double max_phi
 *                const double dv[6]
 *                double p
 *                double q
 *                double r
 *                double Cm_zero
 *                double Cl_alpha
 *                double Cd_zero
 *                double K_Cd
 *                double Cm_alpha
 *                double rho
 *                double V
 *                double S
 *                double wing_chord
 *                double flight_path_angle
 *                double max_alpha
 *                double min_alpha
 *                double Beta
 *                double gamma_quadratic_dv
 *                double gamma_quadratic_du
 *                double gamma_quadratic_wls
 *                double desired_motor_value
 *                double desired_el_value
 *                double desired_az_value
 *                double controller_id
 *                double verbose
 *                double u_out[12]
 *                double residuals[6]
 *                double *elapsed_time
 *                double *N_iterations
 *                double *N_evaluation
 *                double *exitflag
 * Return Type  : void
 */
void Global_controller_fcn_earth_rf_journal(double K_p_T, double K_p_M, double m,
  double I_xx, double I_yy, double I_zz, double l_1, double l_2, double l_3,
  double l_4, double l_z, double Phi, double Theta, double Psi, double Omega_1,
  double Omega_2, double Omega_3, double Omega_4, double b_1, double b_2, double
  b_3, double b_4, double g_1, double g_2, double g_3, double g_4, double
  W_act_motor_const, double W_act_motor_speed, double W_act_tilt_el_const,
  double W_act_tilt_el_speed, double W_act_tilt_az_const, double
  W_act_tilt_az_speed, double W_act_theta_const, double W_act_theta_speed,
  double W_act_phi_const, double W_act_phi_speed, double W_dv_1, double W_dv_2,
  double W_dv_3, double W_dv_4, double W_dv_5, double W_dv_6, double max_omega,
  double min_omega, double max_b, double min_b, double max_g, double min_g,
  double max_theta, double min_theta, double max_phi, const double dv[6], double
  p, double q, double r, double Cm_zero, double Cl_alpha, double Cd_zero, double
  K_Cd, double Cm_alpha, double rho, double V, double S, double wing_chord,
  double flight_path_angle, double max_alpha, double min_alpha, double Beta,
  double gamma_quadratic_dv, double gamma_quadratic_du, double
  gamma_quadratic_wls, double desired_motor_value, double desired_el_value,
  double desired_az_value, double controller_id, double verbose, double u_out[12],
  double residuals[6], double *elapsed_time, double *N_iterations, double
  *N_evaluation, double *exitflag)
{
  b_captured_var dv_global;
  captured_var W_act_motor;
  captured_var W_act_tilt_az;
  captured_var W_act_tilt_el;
  captured_var b_Beta;
  captured_var b_Cd_zero;
  captured_var b_Cl_alpha;
  captured_var b_Cm_alpha;
  captured_var b_Cm_zero;
  captured_var b_I_xx;
  captured_var b_I_yy;
  captured_var b_I_zz;
  captured_var b_K_Cd;
  captured_var b_K_p_M;
  captured_var b_K_p_T;
  captured_var b_Phi;
  captured_var b_Psi;
  captured_var b_S;
  captured_var b_Theta;
  captured_var b_V;
  captured_var b_W_dv_1;
  captured_var b_W_dv_2;
  captured_var b_W_dv_3;
  captured_var b_W_dv_4;
  captured_var b_W_dv_5;
  captured_var b_W_dv_6;
  captured_var b_desired_az_value;
  captured_var b_desired_el_value;
  captured_var b_desired_motor_value;
  captured_var b_flight_path_angle;
  captured_var b_gamma_quadratic_du;
  captured_var b_gamma_quadratic_dv;
  captured_var b_l_1;
  captured_var b_l_2;
  captured_var b_l_3;
  captured_var b_l_4;
  captured_var b_l_z;
  captured_var b_m;
  captured_var b_p;
  captured_var b_q;
  captured_var b_r;
  captured_var b_rho;
  captured_var b_wing_chord;
  captured_var gain_az;
  captured_var gain_el;
  captured_var gain_motor;
  d_struct_T expl_temp;
  double u_out_local_data[144];
  double actual_u[12];
  double u_max[12];
  double u_max_scaled[12];
  double absxk;
  double c_expl_temp;
  double scale;
  double t;
  double u_min_idx_3;
  int b_i;
  int i;
  int i1;
  (void)W_act_theta_const;
  (void)W_act_theta_speed;
  (void)W_act_phi_const;
  (void)W_act_phi_speed;
  (void)max_theta;
  (void)min_theta;
  (void)max_phi;
  (void)min_alpha;
  if (!isInitialized_Global_controller_fcn_earth_rf_journal) {
    Global_controller_fcn_earth_rf_journal_initialize();
  }

  /*  (9e-06, 1.31e-07, 2.35, 0.15, 0.13, 0.2, 0.231, 0.231, 0.39, 0.39, 0, ... */
  /*  0, 0, 0, 100, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0,... */
  /*  1, 0, 1, 0,... */
  /*  1, 0, 0, 0, 0, 0,... */
  /*  0.01, 0.01, 0.01, 0.1, 0.1, 0.01,... */
  /*  900, 100, 25, -130, 45, -45, 80, -30, 40, [0 0 -5 0 0 0]', 0, 0, 0, 0.1, 5.18, 0.38, 0.2, ... */
  /*  -.1, 1.225, 0, 0.55, 0.33, 0, 15, -15, 0, 300, 3e-3, 10000,... */
  /*  0, 0, 0, 1, 1)   */
  /*  (9e-06, 1.31e-07, 2.35, 0.15, 0.13, 0.2, 0.231, 0.231, 0.39, 0.39, 0, 0, 0, 0, 100, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0.01, 0.01, 0.01, 0.1, 0.1, 0.01, 900, 100, 25, -130, 45, -45, 80, -30, 40, [0 0 -5 0 0 0]', 0, 0, 0, 0.1, 5.18, 0.38, 0.2, -.1, 1.225, 0, 0.55, 0.33, 0, 15, -15, 0, 1, 1e-7, 10000, 0, 0, 0, 1, 1) */
  /*     %% Script  */
  /* NONLINEAR CA controller */
  if (controller_id == 1.0) {
    double u_min[12];
    if (desired_motor_value < 120.0) {
      desired_motor_value = (((Omega_1 + Omega_2) + Omega_3) + Omega_4) / 4.0;
    }

    /*  Testing parameters: */
    /*  (9e-06, 1.31e-07, 2.35, 0.15, 0.13, 0.2, 0.231, 0.231, 0.39, 0.39, 0, ... */
    /*  0, 0, 0, 100, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0,... */
    /*  1, 1, 1, 0,... */
    /*  1, 100, 400, -50, 500, -60,... */
    /*  0.01, 0.01, 0.01, 0.1, 0.1, 0.01,... */
    /*  900, 0, 25, -130, 45, -45, 100, -30, 40, [0 0 -5 0 0 0]', 0, 0, 0, 0.1, 5.18, 0.38, 0.2, ... */
    /*  -.1, 1.225, 0, 0.55, 0.33, 0, 15, -15, 0, 100, 1e-3,... */
    /*  0, 0, 0, 0, 0)    */
    b_K_p_T.contents = K_p_T;
    b_K_p_M.contents = K_p_M;
    b_m.contents = m;
    b_I_xx.contents = I_xx;
    b_I_yy.contents = I_yy;
    b_I_zz.contents = I_zz;
    b_l_1.contents = l_1;
    b_l_2.contents = l_2;
    b_l_3.contents = l_3;
    b_l_4.contents = l_4;
    b_l_z.contents = l_z;
    b_Phi.contents = Phi;
    b_Theta.contents = Theta;
    b_Psi.contents = Psi;
    b_W_dv_1.contents = W_dv_1;
    b_W_dv_2.contents = W_dv_2;
    b_W_dv_3.contents = W_dv_3;
    b_W_dv_4.contents = W_dv_4;
    b_W_dv_5.contents = W_dv_5;
    b_W_dv_6.contents = W_dv_6;
    b_p.contents = p;
    b_q.contents = q;
    b_r.contents = r;
    b_Cm_zero.contents = Cm_zero;
    b_Cl_alpha.contents = Cl_alpha;
    b_Cd_zero.contents = Cd_zero;
    b_K_Cd.contents = K_Cd;
    b_Cm_alpha.contents = Cm_alpha;
    b_rho.contents = rho;
    b_V.contents = V;
    b_S.contents = S;
    b_wing_chord.contents = wing_chord;
    b_flight_path_angle.contents = flight_path_angle;
    b_Beta.contents = Beta;
    b_gamma_quadratic_dv.contents = gamma_quadratic_dv;
    b_gamma_quadratic_du.contents = gamma_quadratic_du;
    b_desired_motor_value.contents = desired_motor_value;
    b_desired_el_value.contents = desired_el_value;
    b_desired_az_value.contents = desired_az_value;

    /*  Create variables necessary for the optimization */
    gain_motor.contents = max_omega / 2.0;
    gain_el.contents = (max_b - min_b) * 3.1415926535897931 / 180.0 / 2.0;
    gain_az.contents = (max_g - min_g) * 3.1415926535897931 / 180.0 / 2.0;
    actual_u[0] = Omega_1;
    actual_u[1] = Omega_2;
    actual_u[2] = Omega_3;
    actual_u[3] = Omega_4;
    actual_u[4] = b_1;
    actual_u[5] = b_2;
    actual_u[6] = b_3;
    actual_u[7] = b_4;
    actual_u[8] = g_1;
    actual_u[9] = g_2;
    actual_u[10] = g_3;
    actual_u[11] = g_4;

    /* Build the max and minimum actuator array:  */
    u_max[0] = max_omega;
    u_max[1] = max_omega;
    u_max[2] = max_omega;
    u_max[3] = max_omega;
    u_max[4] = max_b;
    u_max[5] = max_b;
    u_max[6] = max_b;
    u_max[7] = max_b;
    u_max[8] = max_g;
    u_max[9] = max_g;
    u_max[10] = max_g;
    u_max[11] = max_g;
    u_min[0] = min_omega;
    u_min[1] = min_omega;
    u_min[2] = min_omega;
    u_min[3] = min_omega;
    u_min[4] = min_b;
    u_min[5] = min_b;
    u_min[6] = min_b;
    u_min[7] = min_b;
    u_min[8] = min_g;
    u_min[9] = min_g;
    u_min[10] = min_g;
    u_min[11] = min_g;
    for (i = 0; i < 8; i++) {
      u_max[i + 4] = u_max[i + 4] * 3.1415926535897931 / 180.0;
      u_min[i + 4] = u_min[i + 4] * 3.1415926535897931 / 180.0;
    }

    memcpy(&u_max_scaled[0], &u_max[0], 12U * sizeof(double));
    u_max_scaled[0] = u_max[0] / gain_motor.contents;
    u_max_scaled[1] = u_max[1] / gain_motor.contents;
    u_max_scaled[2] = u_max[2] / gain_motor.contents;
    u_max_scaled[3] = u_max[3] / gain_motor.contents;
    scale = u_min[0] / gain_motor.contents;
    absxk = u_min[1] / gain_motor.contents;
    t = u_min[2] / gain_motor.contents;
    u_min_idx_3 = u_min[3] / gain_motor.contents;
    u_min[0] = scale;
    u_min[1] = absxk;
    u_min[2] = t;
    u_min[3] = u_min_idx_3;
    scale = u_max_scaled[4] / gain_el.contents;
    absxk = u_max_scaled[5] / gain_el.contents;
    t = u_max_scaled[6] / gain_el.contents;
    u_min_idx_3 = u_max_scaled[7] / gain_el.contents;
    u_max_scaled[4] = scale;
    u_max_scaled[5] = absxk;
    u_max_scaled[6] = t;
    u_max_scaled[7] = u_min_idx_3;
    scale = u_min[4] / gain_el.contents;
    absxk = u_min[5] / gain_el.contents;
    t = u_min[6] / gain_el.contents;
    u_min_idx_3 = u_min[7] / gain_el.contents;
    u_min[4] = scale;
    u_min[5] = absxk;
    u_min[6] = t;
    u_min[7] = u_min_idx_3;
    scale = u_max_scaled[8] / gain_az.contents;
    absxk = u_max_scaled[9] / gain_az.contents;
    t = u_max_scaled[10] / gain_az.contents;
    u_min_idx_3 = u_max_scaled[11] / gain_az.contents;
    u_max_scaled[8] = scale;
    u_max_scaled[9] = absxk;
    u_max_scaled[10] = t;
    u_max_scaled[11] = u_min_idx_3;
    scale = u_min[8] / gain_az.contents;
    absxk = u_min[9] / gain_az.contents;
    t = u_min[10] / gain_az.contents;
    u_min_idx_3 = u_min[11] / gain_az.contents;
    u_min[8] = scale;
    u_min[9] = absxk;
    u_min[10] = t;
    u_min[11] = u_min_idx_3;
    memcpy(&u_max[0], &actual_u[0], 12U * sizeof(double));
    u_max[0] = Omega_1 / gain_motor.contents;
    u_max[1] = Omega_2 / gain_motor.contents;
    u_max[2] = Omega_3 / gain_motor.contents;
    u_max[3] = Omega_4 / gain_motor.contents;
    scale = u_max[4] / gain_el.contents;
    absxk = u_max[5] / gain_el.contents;
    t = u_max[6] / gain_el.contents;
    u_min_idx_3 = u_max[7] / gain_el.contents;
    u_max[4] = scale;
    u_max[5] = absxk;
    u_max[6] = t;
    u_max[7] = u_min_idx_3;
    scale = u_max[8] / gain_az.contents;
    absxk = u_max[9] / gain_az.contents;
    t = u_max[10] / gain_az.contents;
    u_min_idx_3 = u_max[11] / gain_az.contents;
    u_max[8] = scale;
    u_max[9] = absxk;
    u_max[10] = t;
    u_max[11] = u_min_idx_3;

    /*  Apply Nonlinear optimization algorithm: */
    c_compute_acc_nonlinear_earth_r(actual_u, b_Theta.contents, b_Phi.contents,
      b_Psi.contents, b_p.contents, b_q.contents, b_r.contents, b_K_p_T.contents,
      b_K_p_M.contents, b_m.contents, b_I_xx.contents, b_I_yy.contents,
      b_I_zz.contents, b_l_1.contents, b_l_2.contents, b_l_3.contents,
      b_l_4.contents, b_l_z.contents, b_Cl_alpha.contents, b_Cd_zero.contents,
      b_K_Cd.contents, b_Cm_alpha.contents, b_Cm_zero.contents, b_rho.contents,
      b_V.contents, b_S.contents, b_wing_chord.contents,
      b_flight_path_angle.contents, b_Beta.contents, dv_global.contents);
    for (i = 0; i < 6; i++) {
      dv_global.contents[i] += dv[i];
    }

    char b_expl_temp[3];

    /* Compute weights for actuators and make sure they are always positive */
    scale = W_act_motor_const + W_act_motor_speed * b_V.contents;
    W_act_motor.contents = fmax(0.0, scale);
    scale = W_act_tilt_el_const + W_act_tilt_el_speed * b_V.contents;
    W_act_tilt_el.contents = fmax(0.0, scale);
    scale = W_act_tilt_az_const + W_act_tilt_az_speed * b_V.contents;
    W_act_tilt_az.contents = fmax(0.0, scale);

    /* Default values for the optimizer: */
    tic();
    expl_temp.W_dv_4 = &b_W_dv_4;
    expl_temp.l_2 = &b_l_2;
    expl_temp.l_1 = &b_l_1;
    expl_temp.q = &b_q;
    expl_temp.W_dv_6 = &b_W_dv_6;
    expl_temp.Cm_alpha = &b_Cm_alpha;
    expl_temp.l_3 = &b_l_3;
    expl_temp.l_4 = &b_l_4;
    expl_temp.wing_chord = &b_wing_chord;
    expl_temp.Cm_zero = &b_Cm_zero;
    expl_temp.K_p_M = &b_K_p_M;
    expl_temp.l_z = &b_l_z;
    expl_temp.I_yy = &b_I_yy;
    expl_temp.I_xx = &b_I_xx;
    expl_temp.r = &b_r;
    expl_temp.p = &b_p;
    expl_temp.I_zz = &b_I_zz;
    expl_temp.W_dv_5 = &b_W_dv_5;
    expl_temp.W_dv_1 = &b_W_dv_1;
    expl_temp.W_dv_3 = &b_W_dv_3;
    expl_temp.desired_az_value = &b_desired_az_value;
    expl_temp.W_act_tilt_az = &W_act_tilt_az;
    expl_temp.desired_el_value = &b_desired_el_value;
    expl_temp.W_act_tilt_el = &W_act_tilt_el;
    expl_temp.m = &b_m;
    expl_temp.gain_az = &gain_az;
    expl_temp.gain_el = &gain_el;
    expl_temp.K_p_T = &b_K_p_T;
    expl_temp.Phi = &b_Phi;
    expl_temp.Psi = &b_Psi;
    expl_temp.Cd_zero = &b_Cd_zero;
    expl_temp.Cl_alpha = &b_Cl_alpha;
    expl_temp.K_Cd = &b_K_Cd;
    expl_temp.flight_path_angle = &b_flight_path_angle;
    expl_temp.Theta = &b_Theta;
    expl_temp.Beta = &b_Beta;
    expl_temp.rho = &b_rho;
    expl_temp.V = &b_V;
    expl_temp.S = &b_S;
    expl_temp.dv_global = &dv_global;
    expl_temp.gamma_quadratic_dv = &b_gamma_quadratic_dv;
    expl_temp.W_dv_2 = &b_W_dv_2;
    expl_temp.gain_motor = &gain_motor;
    expl_temp.desired_motor_value = &b_desired_motor_value;
    expl_temp.gamma_quadratic_du = &b_gamma_quadratic_du;
    expl_temp.W_act_motor = &W_act_motor;
    fmincon(&expl_temp, u_max, u_min, u_max_scaled, u_out, &u_min_idx_3,
            exitflag, N_iterations, N_evaluation, b_expl_temp, &scale, &absxk,
            &t, &c_expl_temp);
    *elapsed_time = toc();
    scale = gain_motor.contents;
    u_out[0] *= scale;
    u_out[1] *= scale;
    u_out[2] *= scale;
    u_out[3] *= scale;
    scale = gain_el.contents;
    u_out[4] *= scale;
    u_out[5] *= scale;
    u_out[6] *= scale;
    u_out[7] *= scale;
    scale = gain_az.contents;
    u_out[8] *= scale;
    u_out[9] *= scale;
    u_out[10] *= scale;
    u_out[11] *= scale;
    c_compute_acc_nonlinear_earth_r(u_out, b_Theta.contents, b_Phi.contents,
      b_Psi.contents, b_p.contents, b_q.contents, b_r.contents, b_K_p_T.contents,
      b_K_p_M.contents, b_m.contents, b_I_xx.contents, b_I_yy.contents,
      b_I_zz.contents, b_l_1.contents, b_l_2.contents, b_l_3.contents,
      b_l_4.contents, b_l_z.contents, b_Cl_alpha.contents, b_Cd_zero.contents,
      b_K_Cd.contents, b_Cm_alpha.contents, b_Cm_zero.contents, b_rho.contents,
      b_V.contents, b_S.contents, b_wing_chord.contents,
      b_flight_path_angle.contents, b_Beta.contents, residuals);
    for (i = 0; i < 6; i++) {
      residuals[i] = dv_global.contents[i] - residuals[i];
    }

    memcpy(&u_out_local_data[0], &u_out[0], 12U * sizeof(double));

    /* WLS CA controller */
  } else if (controller_id == 2.0) {
    double b_dv[6];
    if (desired_motor_value < 120.0) {
      desired_motor_value = (((Omega_1 + Omega_2) + Omega_3) + Omega_4) / 4.0;
    }

    /*          [u_out_local, residuals, elapsed_time, N_iterations, N_evaluation, exitflag]  = WLS_controller_fcn_earth_rf_journal(K_p_T, K_p_M, m, I_xx, I_yy, I_zz, l_1, l_2, l_3, l_4, l_z, ... */
    /*                                                   Phi, Theta, Psi, Omega_1, Omega_2, Omega_3, Omega_4, b_1, b_2, b_3, b_4, g_1, g_2, g_3, g_4, ... */
    /*                                                   W_act_motor_const, W_act_motor_speed, W_act_tilt_el_const, W_act_tilt_el_speed, ... */
    /*                                                   W_act_tilt_az_const, W_act_tilt_az_speed, W_act_theta_const, W_act_theta_speed, W_act_phi_const, W_act_phi_speed, ... */
    /*                                                   W_dv_1, W_dv_2, W_dv_3, W_dv_4, W_dv_5, W_dv_6, ... */
    /*                                                   max_omega, min_omega, max_b, min_b, max_g, min_g, max_theta, min_theta, max_phi, dv, p, q, r, Cm_zero, Cl_alpha, Cd_zero, K_Cd,... */
    /*                                                   Cm_alpha, rho, V, S, wing_chord, flight_path_angle, max_alpha, min_alpha, Beta, gamma_quadratic_wls, ... */
    /*                                                   desired_motor_value, desired_el_value, desired_az_value, 0, 0, max_iter_wls); */
    /*          [u_out_local, residuals, elapsed_time, N_iterations, N_evaluation, exitflag]  = WLS_controller_fcn_earth_rf_journal_u_global(K_p_T, K_p_M, m, I_xx, I_yy, I_zz, l_1, l_2, l_3, l_4, l_z, ... */
    /*                                                   Phi, Theta, Psi, Omega_1, Omega_2, Omega_3, Omega_4, b_1, b_2, b_3, b_4, g_1, g_2, g_3, g_4, ... */
    /*                                                   W_act_motor_const, W_act_motor_speed, W_act_tilt_el_const, W_act_tilt_el_speed, ... */
    /*                                                   W_act_tilt_az_const, W_act_tilt_az_speed, W_act_theta_const, W_act_theta_speed, W_act_phi_const, W_act_phi_speed, ... */
    /*                                                   W_dv_1, W_dv_2, W_dv_3, W_dv_4, W_dv_5, W_dv_6, ... */
    /*                                                   max_omega, min_omega, max_b, min_b, max_g, min_g, max_theta, min_theta, max_phi, dv, p, q, r, Cm_zero, Cl_alpha, Cd_zero, K_Cd,... */
    /*                                                   Cm_alpha, rho, V, S, wing_chord, flight_path_angle, max_alpha, min_alpha, Beta, gamma_quadratic_wls, ... */
    /*                                                   desired_motor_value, desired_el_value, desired_az_value, 0, 0, max_iter_wls);     */
    for (i1 = 0; i1 < 6; i1++) {
      b_dv[i1] = dv[i1];
    }

    c_WLS_controller_fcn_earth_rf_j(K_p_T, K_p_M, m, I_xx, I_yy, I_zz, l_1, l_2,
      l_3, l_4, l_z, Phi, Theta, Psi, Omega_1, Omega_2, Omega_3, Omega_4, b_1,
      b_2, b_3, b_4, g_1, g_2, g_3, g_4, W_act_motor_const, W_act_motor_speed,
      W_act_tilt_el_const, W_act_tilt_el_speed, W_act_tilt_az_const,
      W_act_tilt_az_speed, W_dv_1, W_dv_2, W_dv_3, W_dv_4, W_dv_5, W_dv_6,
      max_omega, min_omega, max_b, min_b, max_g, min_g, b_dv, p, q, r, Cm_zero,
      Cl_alpha, Cd_zero, K_Cd, Cm_alpha, rho, V, S, wing_chord,
      flight_path_angle, max_alpha, Beta, gamma_quadratic_wls,
      desired_motor_value, desired_el_value, desired_az_value, u_max, residuals,
      elapsed_time, N_iterations, N_evaluation, exitflag);
    memcpy(&u_out_local_data[0], &u_max[0], 12U * sizeof(double));

    /* PINV controller */
  } else if (controller_id == 3.0) {
    double b_dv[6];
    if (desired_motor_value < 120.0) {
      desired_motor_value = (((Omega_1 + Omega_2) + Omega_3) + Omega_4) / 4.0;
    }

    for (b_i = 0; b_i < 6; b_i++) {
      b_dv[b_i] = dv[b_i];
    }

    c_Basic_inversion_controller_fc(K_p_T, K_p_M, m, I_xx, I_yy, I_zz, l_1, l_2,
      l_3, l_4, l_z, Phi, Theta, Psi, Omega_1, Omega_2, Omega_3, Omega_4, b_1,
      b_2, b_3, b_4, g_1, g_2, g_3, g_4, max_omega, min_omega, max_b, min_b,
      max_g, min_g, b_dv, p, q, r, Cm_zero, Cl_alpha, Cd_zero, K_Cd, Cm_alpha,
      rho, V, S, wing_chord, flight_path_angle, Beta, desired_motor_value,
      desired_el_value, desired_az_value, u_max, residuals, elapsed_time,
      exitflag);
    memcpy(&u_out_local_data[0], &u_max[0], 12U * sizeof(double));
    *N_iterations = 1.0;
    *N_evaluation = 1.0;
  } else {
    memset(&u_out_local_data[0], 0, 12U * sizeof(double));
    for (i = 0; i < 6; i++) {
      residuals[i] = 0.0;
    }

    *elapsed_time = 0.0;
    *N_iterations = 0.0;
    *N_evaluation = 0.0;
    *exitflag = -10.0;
  }

  memcpy(&u_out[0], &u_out_local_data[0], 12U * sizeof(double));

  /*  Print infos */
  if (verbose != 0.0) {
    printf("\n Solution = ");
    fflush(stdout);
    for (i = 0; i < 12; i++) {
      printf(" %f ", u_out_local_data[i]);
      fflush(stdout);
    }

    printf("\n");
    fflush(stdout);
    printf("\n Elapsed time = %f \n", *elapsed_time);
    fflush(stdout);
    printf("\n Number of iterations = %f \n", *N_iterations);
    fflush(stdout);
    printf("\n Number of evaluation = %f \n", *N_evaluation);
    fflush(stdout);
    printf("\n Residuals = ");
    fflush(stdout);
    for (i = 0; i < 6; i++) {
      printf(" %f ", residuals[i]);
      fflush(stdout);
    }

    printf("\n");
    fflush(stdout);
    u_min_idx_3 = 0.0;
    scale = 3.3121686421112381E-170;
    for (i = 0; i < 6; i++) {
      absxk = fabs(residuals[i]);
      if (absxk > scale) {
        t = scale / absxk;
        u_min_idx_3 = u_min_idx_3 * t * t + 1.0;
        scale = absxk;
      } else {
        t = absxk / scale;
        u_min_idx_3 += t * t;
      }
    }

    printf("\n Residual norm = %f \n", scale * sqrt(u_min_idx_3));
    fflush(stdout);
    memcpy(&u_max[0], &u_out[0], 12U * sizeof(double));
    scale = max_omega / 2.0;
    absxk = (max_b - min_b) * 3.1415926535897931 / 180.0 / 2.0;
    t = (max_g - min_g) * 3.1415926535897931 / 180.0 / 2.0;
    u_max[0] = u_out[0] / scale;
    u_max[4] /= absxk;
    u_max[8] /= t;
    u_max[1] = u_out[1] / scale;
    u_max[5] /= absxk;
    u_max[9] /= t;
    u_max[2] = u_out[2] / scale;
    u_max[6] /= absxk;
    u_max[10] /= t;
    u_max[3] = u_out[3] / scale;
    u_max[7] /= absxk;
    u_max[11] /= t;
    u_min_idx_3 = 0.0;
    scale = 3.3121686421112381E-170;
    for (i = 0; i < 12; i++) {
      absxk = fabs(u_max[i]);
      if (absxk > scale) {
        t = scale / absxk;
        u_min_idx_3 = u_min_idx_3 * t * t + 1.0;
        scale = absxk;
      } else {
        t = absxk / scale;
        u_min_idx_3 += t * t;
      }
    }

    printf("\n Solution scaled norm = %f \n", scale * sqrt(u_min_idx_3));
    fflush(stdout);
    printf("\n Exit flag optimizer = %f \n", *exitflag);
    fflush(stdout);
  }
}

/*
 * Arguments    : void
 * Return Type  : void
 */
void Global_controller_fcn_earth_rf_journal_initialize(void)
{
  omp_init_nest_lock(&Global_controller_fcn_earth_rf_journal_nestLockGlobal);
  savedTime_not_empty = false;
  freq_not_empty = false;
  isInitialized_Global_controller_fcn_earth_rf_journal = true;
}

/*
 * Arguments    : void
 * Return Type  : void
 */
void Global_controller_fcn_earth_rf_journal_terminate(void)
{
  omp_destroy_nest_lock(&Global_controller_fcn_earth_rf_journal_nestLockGlobal);
  isInitialized_Global_controller_fcn_earth_rf_journal = false;
}

/*
 * File trailer for Global_controller_fcn_earth_rf_journal.c
 *
 * [EOF]
 */
