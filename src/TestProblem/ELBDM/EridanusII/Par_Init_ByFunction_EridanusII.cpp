#include "GAMER.h"

#ifdef PARTICLE

#ifdef SUPPORT_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#else
#error : ERROR : please turn on SUPPORT_GSL for the EridanusII test problem !!
#endif

extern double Soliton_CoreRadius;

extern int    Star_RSeed;
extern int    Star_SigmaMode;
extern double Star_Rho0;
extern double Star_R0;
extern double Star_MaxR;
extern double Star_Center[3];
extern int    Star_MassProfNBin;

static RandomNumber_t *RNG = NULL;


static double MassProf_Star( const double r );
static void   RanVec_FixRadius( const double r, double RanVec[] );




//-------------------------------------------------------------------------------------------------------
// Function    :  Par_Init_ByFunction_EridanusII
// Description :  User-specified function to initialize particle attributes
//
// Note        :  1. Invoked by Init_GAMER() using the function pointer "Par_Init_ByFunction_Ptr"
//                   --> This function pointer may be reset by various test problem initializers, in which case
//                       this funtion will become useless
//                2. Periodicity should be taken care of in this function
//                   --> No particles should lie outside the simulation box when the periodic BC is adopted
//                   --> However, if the non-periodic BC is adopted, particles are allowed to lie outside the box
//                       (more specifically, outside the "active" region defined by amr->Par->RemoveCell)
//                       in this function. They will later be removed automatically when calling Par_Aux_InitCheck()
//                       in Init_GAMER().
//                3. Particles set by this function are only temporarily stored in this MPI rank
//                   --> They will later be redistributed when calling Par_FindHomePatch_UniformGrid()
//                       and LB_Init_LoadBalance()
//                   --> Therefore, there is no constraint on which particles should be set by this function
//
// Parameter   :  NPar_ThisRank : Number of particles to be set by this MPI rank
//                NPar_AllRank  : Total Number of particles in all MPI ranks
//                ParMass       : Particle mass     array with the size of NPar_ThisRank
//                ParPosX/Y/Z   : Particle position array with the size of NPar_ThisRank
//                ParVelX/Y/Z   : Particle velocity array with the size of NPar_ThisRank
//                ParTime       : Particle time     array with the size of NPar_ThisRank
//                AllAttribute  : Pointer array for all particle attributes
//                                --> Dimension = [PAR_NATT_TOTAL][NPar_ThisRank]
//                                --> Use the attribute indices defined in Field.h (e.g., Idx_ParCreTime)
//                                    to access the data
//
// Return      :  ParMass, ParPosX/Y/Z, ParVelX/Y/Z, ParTime, AllAttribute
//-------------------------------------------------------------------------------------------------------
void Par_Init_ByFunction_EridanusII( const long NPar_ThisRank, const long NPar_AllRank,
                                  real *ParMass, real *ParPosX, real *ParPosY, real *ParPosZ,
                                  real *ParVelX, real *ParVelY, real *ParVelZ, real *ParTime,
                                  real *ParType, real *AllAttribute[PAR_NATT_TOTAL] )
{

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ...\n", __FUNCTION__ );

   
   real *Mass_AllRank   = NULL;
   real *Pos_AllRank[3] = { NULL, NULL, NULL };
   real *Vel_AllRank[3] = { NULL, NULL, NULL };

// only the master rank will construct the initial condition
   if ( MPI_Rank == 0 )
   {
      const double TotM_Inf    = 4.0/3.0*M_PI*CUBE(Star_R0)*Star_Rho0;
      const double Vmax_Fac    = sqrt( 2.0*NEWTON_G*TotM_Inf );                     // for SigmaMode==0

      const double m22         = ELBDM_MASS*UNIT_M/(Const_eV/SQR(Const_c))/1.0e-22;
      const double rc_kpc      = Soliton_CoreRadius*UNIT_L/Const_kpc;
      const double peak_rho    = 1.945e7/SQR( m22*rc_kpc*rc_kpc )*Const_Msun/CUBE(Const_kpc)/(UNIT_M/CUBE(UNIT_L));
      const double Sigma_Fac   = 4.0*M_PI/9.0*NEWTON_G*peak_rho*SQR(Star_R0);       // for SigmaMode==1

      double *Table_MassProf_r = NULL;
      double *Table_MassProf_M = NULL;
      double  TotM, ParM, dr, RanM, RanR, EstM, ErrM, ErrM_Max=-1.0, RanVec[3];
      double  Vmax, RanV, RanProb, Prob, Sigma;

//    initialize GSL random number generator for SigmaMode==1
      gsl_rng *GSL_RNG = NULL;
      if ( Star_SigmaMode == 1 )
      {
         GSL_RNG = gsl_rng_alloc( gsl_rng_mt19937 );
         gsl_rng_set( GSL_RNG, Star_RSeed );
      }

      Mass_AllRank = new real [NPar_AllRank];
      for (int d=0; d<3; d++)
      {
         Pos_AllRank[d] = new real [NPar_AllRank];
         Vel_AllRank[d] = new real [NPar_AllRank];
      }


//    initialize the random number generator
      RNG = new RandomNumber_t( 1 );
      RNG->SetSeed( 0, Star_RSeed );


//    determine the total enclosed mass within the maximum radius
      TotM = MassProf_Star( Star_MaxR );
      ParM = TotM / NPar_AllRank;

//    construct the mass profile table
      Table_MassProf_r = new double [Star_MassProfNBin];
      Table_MassProf_M = new double [Star_MassProfNBin];

      dr = Star_MaxR / (Star_MassProfNBin-1);

      for (int b=0; b<Star_MassProfNBin; b++)
      {
         Table_MassProf_r[b] = dr*b;
         Table_MassProf_M[b] = MassProf_Star( Table_MassProf_r[b] );
      }


//    set particle attributes
      for (long p=0; p<NPar_AllRank; p++)
      {
//       mass
         Mass_AllRank[p] = ParM;


//       position
//       --> sample from the cumulative mass profile with linear interpolation
         RanM = RNG->GetValue( 0, 0.0, 1.0 )*TotM;
         RanR = Mis_InterpolateFromTable( Star_MassProfNBin, Table_MassProf_M, Table_MassProf_r, RanM );

//       record the maximum error
         EstM     = MassProf_Star( RanR );
         ErrM     = fabs( (EstM-RanM)/RanM );
         ErrM_Max = fmax( ErrM, ErrM_Max );

//       randomly set the position vector with a given radius
         RanVec_FixRadius( RanR, RanVec );
         for (int d=0; d<3; d++)    Pos_AllRank[d][p] = RanVec[d] + Star_Center[d];

//       check periodicity
         for (int d=0; d<3; d++)
         {
            if ( OPT__BC_FLU[d*2] == BC_FLU_PERIODIC )
               Pos_AllRank[d][p] = FMOD( Pos_AllRank[d][p]+(real)amr->BoxSize[d], (real)amr->BoxSize[d] );
         }


//       velocity
//       mode 0: stars are self-bound
         if      ( Star_SigmaMode == 0 )
         {
//          determine the maximum velocity (i.e., the escaping velocity)
            Vmax = Vmax_Fac*pow( SQR(Star_R0) + SQR(RanR), -0.25 );

//          randomly determine the velocity amplitude (ref: Aarseth, S. et al. 1974, A&A, 37, 183: Eq. [A4,A5])
            do
            {
               RanV    = RNG->GetValue( 0, 0.0, 1.0 );         // (0.0, 1.0)
               RanProb = RNG->GetValue( 0, 0.0, 0.1 );         // (0.0, 0.1)
               Prob    = SQR(RanV)*pow( 1.0-SQR(RanV), 3.5 );  // < 0.1
            }
            while ( RanProb > Prob );

            RanVec_FixRadius( RanV*Vmax, RanVec );
         }

//       mode 1: soliton-dominated and stars are well within the soliton radius
         else if ( Star_SigmaMode == 1 )
         {
//          determine the velocity dispersion
            Sigma = sqrt( Sigma_Fac*(1.0+SQR(RanR/Star_R0)) );

//          randomly determine the velocity amplitude
//          --> assume velocity distribution is isotropic and Gaussian with a standard deviation of Sigma
            for (int d=0; d<3; d++)    RanVec[d] = gsl_ran_gaussian( GSL_RNG, Sigma );
         }

         else
            Aux_Error( ERROR_INFO, "unsupported Star_SigmaMode !!\n" );

//       store the velocity
         for (int d=0; d<3; d++)    Vel_AllRank[d][p] = RanVec[d];

      } // for (long p=0; p<NPar_AllRank; p++)


//    remove the center-of-mass velocity
      double Vcm[3] = { 0.0, 0.0, 0.0 };

      for (long p=0; p<NPar_AllRank; p++)
      for (int d=0; d<3; d++)
         Vcm[d] += ParM*Vel_AllRank[d][p];

      for (int d=0; d<3; d++)
         Vcm[d] /= TotM;

      for (long p=0; p<NPar_AllRank; p++)
      for (int d=0; d<3; d++)
         Vel_AllRank[d][p] -= Vcm[d];


      Aux_Message( stdout, "   Total enclosed mass within MaxR  = %13.7e\n",  TotM );
      Aux_Message( stdout, "   Total enclosed mass to inifinity = %13.7e\n",  TotM_Inf );
      Aux_Message( stdout, "   Enclosed mass ratio              = %6.2f%%\n", 100.0*TotM/TotM_Inf );
      Aux_Message( stdout, "   Particle mass                    = %13.7e\n",  ParM );
      Aux_Message( stdout, "   Maximum mass interpolation error = %13.7e\n",  ErrM_Max );


//    free memory
      delete [] Table_MassProf_r;
      delete [] Table_MassProf_M;

      if ( Star_SigmaMode == 1 )    gsl_rng_free( GSL_RNG );
   } // if ( MPI_Rank == 0 )


// synchronize all particles to the physical time on the base level
   for (long p=0; p<NPar_ThisRank; p++)   {ParTime[p] = Time[0]; ParType[p] = PTYPE_GENERIC_MASSIVE  ;}


// get the number of particles in each rank and set the corresponding offsets
   if ( NPar_AllRank > (long)__INT_MAX__ )
      Aux_Error( ERROR_INFO, "NPar_Active_AllRank (%ld) exceeds the maximum integer (%ld) --> MPI will likely fail !!\n",
                 NPar_AllRank, (long)__INT_MAX__ );

   int NSend[MPI_NRank], SendDisp[MPI_NRank];
   int NPar_ThisRank_int = NPar_ThisRank;    // (i) convert to "int" and (ii) remove the "const" declaration
                                             // --> (ii) is necessary for OpenMPI version < 1.7

   MPI_Gather( &NPar_ThisRank_int, 1, MPI_INT, NSend, 1, MPI_INT, 0, MPI_COMM_WORLD );

   if ( MPI_Rank == 0 )
   {
      SendDisp[0] = 0;
      for (int r=1; r<MPI_NRank; r++)  SendDisp[r] = SendDisp[r-1] + NSend[r-1];
   }


// send particle attributes from the master rank to all ranks
   real *Mass   =   ParMass;
   real *Pos[3] = { ParPosX, ParPosY, ParPosZ };
   real *Vel[3] = { ParVelX, ParVelY, ParVelZ };

#  ifdef FLOAT8
   MPI_Scatterv( Mass_AllRank, NSend, SendDisp, MPI_DOUBLE, Mass, NPar_ThisRank, MPI_DOUBLE, 0, MPI_COMM_WORLD );

   for (int d=0; d<3; d++)
   {
      MPI_Scatterv( Pos_AllRank[d], NSend, SendDisp, MPI_DOUBLE, Pos[d], NPar_ThisRank, MPI_DOUBLE, 0, MPI_COMM_WORLD );
      MPI_Scatterv( Vel_AllRank[d], NSend, SendDisp, MPI_DOUBLE, Vel[d], NPar_ThisRank, MPI_DOUBLE, 0, MPI_COMM_WORLD );
   }

#  else
   MPI_Scatterv( Mass_AllRank, NSend, SendDisp, MPI_FLOAT,  Mass, NPar_ThisRank, MPI_FLOAT,  0, MPI_COMM_WORLD );

   for (int d=0; d<3; d++)
   {
      MPI_Scatterv( Pos_AllRank[d], NSend, SendDisp, MPI_FLOAT,  Pos[d], NPar_ThisRank, MPI_FLOAT,  0, MPI_COMM_WORLD );
      MPI_Scatterv( Vel_AllRank[d], NSend, SendDisp, MPI_FLOAT,  Vel[d], NPar_ThisRank, MPI_FLOAT,  0, MPI_COMM_WORLD );
   }
#  endif


   if ( MPI_Rank == 0 )
   {
      delete RNG;
      delete [] Mass_AllRank;

      for (int d=0; d<3; d++)
      {
         delete [] Pos_AllRank[d];
         delete [] Vel_AllRank[d];
      }
   }


   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ... done\n", __FUNCTION__ );

} // FUNCTION : Par_Init_ByFunction_EridanusII



//-------------------------------------------------------------------------------------------------------
// Function    :  MassProf_Star
// Description :  Mass profile of the star cluster (using the Plummer model for now) 
//
// Note        :  Calculate the enclosed mass within the given radius
//
// Parameter   :  r : Input radius
//
// Return      :  Enclosed mass
//-------------------------------------------------------------------------------------------------------
double MassProf_Star( const double r )
{

   const double x = r / Star_R0;

   return 4.0/3.0*M_PI*Star_Rho0*CUBE(r)*pow( 1.0+x*x, -1.5 );

} // FUNCTION : MassProf_Star



//-------------------------------------------------------------------------------------------------------
// Function    :  RanVec_FixRadius
// Description :  Compute a random 3D vector with a fixed radius
//
// Note        :  Uniformly random sample in theta and phi does NOT give a uniformly random sample in 3D space
//                --> Uniformly random sample in a 3D sphere and then normalize all vectors to the given radius
//
// Parameter   :  r      : Input radius
//                RanVec : Array to store the random 3D vector
//
// Return      :  RanVec
//-------------------------------------------------------------------------------------------------------
void RanVec_FixRadius( const double r, double RanVec[] )
{

   double Norm, RanR2;

   do
   {
      RanR2 = 0.0;

      for (int d=0; d<3; d++)
      {
         RanVec[d]  = RNG->GetValue( 0, -1.0, +1.0 );
         RanR2     += SQR( RanVec[d] );
      }
   } while ( RanR2 > 1.0 );

   Norm = r / sqrt( RanR2 );

   for (int d=0; d<3; d++)    RanVec[d] *= Norm;

} // FUNCTION : RanVec_FixRadius



#endif // #ifdef PARTICLE
