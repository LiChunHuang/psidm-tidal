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

extern bool   Star_AddParForRestart;
extern long   Star_AddParForRestart_NPar;
extern double Star_AddParForRestart_PeakRho;

static RandomNumber_t *RNG = NULL;


static double MassProf_Star( const double r );
static void   RanVec_FixRadius( const double r, double RanVec[] );

extern void (*Aux_Record_User_Ptr)();
void SetExtPotAuxArray_EridanusII( double AuxArray_Flt[], int AuxArray_Int[] );

extern double *Eridanus_Prof;
extern int     Eridanus_Prof_NBin;


       real *h_ExtPotGrep_Eri;
extern bool Tidal_Orbit_Type;

#ifdef GPU
extern void SetGPUPtr( const real *h_table);
#endif


//-------------------------------------------------------------------------------------------------------
// Function    :  Init_User_EridanusII
// Description :  User-specified initialization
//
// Note        :  1. Add particles after restart
//                2. Set the central coordinates of tidal field
//
// Parameter   :  None
//
// Return      :  ParMass, ParPosX/Y/Z, ParVelX/Y/Z, ParTime
//-------------------------------------------------------------------------------------------------------
void Init_User_EridanusII()
{
    double Tidal_Orbit = (Tidal_Orbit_Type) ? +1.0 : -1.0 ;   
    const bool Orbit = (Tidal_Orbit>0.0) ? true : false;
    
    if(!Orbit)
    {
       h_ExtPotGrep_Eri = (real*)malloc(Eridanus_Prof_NBin*3*sizeof(real));
//     h_ExtPotGrep_Eri = new real [Eridanus_Prof_NBin*3];

       double  *Prof_T     = Eridanus_Prof + 0*Eridanus_Prof_NBin;
       double  *Prof_R     = Eridanus_Prof + 1*Eridanus_Prof_NBin;
       double  *Prof_theta = Eridanus_Prof + 2*Eridanus_Prof_NBin; 
 
       for (int b=0; b<Eridanus_Prof_NBin; b++)
       {
          h_ExtPotGrep_Eri[b                     ] = Prof_T[b];
          h_ExtPotGrep_Eri[b+1*Eridanus_Prof_NBin] = Prof_R[b];
          h_ExtPotGrep_Eri[b+2*Eridanus_Prof_NBin] = Prof_theta[b];
       }
 
       h_ExtPotGenePtr[0] = (real**)(h_ExtPotGrep_Eri);
       h_ExtPotGenePtr[1] = (real**)(h_ExtPotGrep_Eri+1*Eridanus_Prof_NBin);
       h_ExtPotGenePtr[2] = (real**)(h_ExtPotGrep_Eri+2*Eridanus_Prof_NBin);
     
#    ifdef GPU
 
       SetGPUPtr(h_ExtPotGrep_Eri);
#    endif 

   }





   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ...\n", __FUNCTION__ );


// 1. add particles after restart
   if ( amr->Par->Init == PAR_INIT_BY_RESTART  &&  OPT__RESTART_RESET  &&  Star_AddParForRestart )
   {
      if ( MPI_Rank == 0 )    Aux_Message( stdout, "Adding particles after restart ...\n" );

// check
   if ( Star_AddParForRestart_NPar < 0 )
      Aux_Error( ERROR_INFO, "Star_AddParForRestart_NPar = %ld < 0 !!\n", Star_AddParForRestart_NPar );

   if ( Star_SigmaMode == 1  &&  Star_AddParForRestart_PeakRho <= 0.0 )
      Aux_Error( ERROR_INFO, "Star_AddParForRestart_PeakRho = %14.7e < 0.0 !!\n", Star_AddParForRestart_PeakRho );


   const long   NNewPar        = ( MPI_Rank == 0 ) ? Star_AddParForRestart_NPar : 0;
   const long   NPar_AllRank   = NNewPar;

   real *NewParAtt[PAR_NATT_TOTAL];

   for (int v=0; v<PAR_NATT_TOTAL; v++)   NewParAtt[v] = new real [NNewPar];


// set particle attributes
// ============================================================================================================
   real *Time_AllRank   = NewParAtt[PAR_TIME];
   real *Mass_AllRank   = NewParAtt[PAR_MASS];
   real *Pos_AllRank[3] = { NewParAtt[PAR_POSX], NewParAtt[PAR_POSY], NewParAtt[PAR_POSZ] };
   real *Vel_AllRank[3] = { NewParAtt[PAR_VELX], NewParAtt[PAR_VELY], NewParAtt[PAR_VELZ] };

// only the master rank will construct the initial condition
   if ( MPI_Rank == 0 )
   {
      const double TotM_Inf    = 4.0/3.0*M_PI*CUBE(Star_R0)*Star_Rho0;
      const double Vmax_Fac    = sqrt( 2.0*NEWTON_G*TotM_Inf );                                       // for SigmaMode==0

      const double Sigma_Fac   = 4.0*M_PI/9.0*NEWTON_G*Star_AddParForRestart_PeakRho*SQR(Star_R0);    // for SigmaMode==1

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
//       time
         Time_AllRank[p] = Time[0];


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
      delete RNG;
      delete [] Table_MassProf_r;
      delete [] Table_MassProf_M;

      if ( Star_SigmaMode == 1 )    gsl_rng_free( GSL_RNG );

   } // if ( MPI_Rank == 0 )


// add particles here
   Par_AddParticleAfterInit( NNewPar, NewParAtt );


// free memory
   for (int v=0; v<PAR_NATT_TOTAL; v++)   delete [] NewParAtt[v];


// refine the grids
   const bool   Redistribute_Yes = true;
   const bool   SendGridData_Yes = true;
   const bool   ResetLB_Yes      = true;
#  if ( defined PARTICLE  &&  defined LOAD_BALANCE )
   const double Par_Weight       = amr->LB->Par_Weight;
#  else
   const double Par_Weight       = 0.0;
#  endif
#  ifdef LOAD_BALANCE
   const UseLBFunc_t UseLB       = USELB_YES;
#  else
   const UseLBFunc_t UseLB       = USELB_NO;
#  endif

   for (int lv=0; lv<MAX_LEVEL; lv++)
   {
      if ( MPI_Rank == 0 )    Aux_Message( stdout, "   Refining level %d ... ", lv );

      Flag_Real( lv, UseLB );

      Refine( lv, UseLB );

#     ifdef LOAD_BALANCE
//    no need to exchange potential since we haven't calculated it yet
      Buf_GetBufferData( lv,   amr->FluSg[lv  ], NULL_INT, NULL_INT, DATA_AFTER_REFINE, _TOTAL, _NONE, Flu_ParaBuf, USELB_YES );

      Buf_GetBufferData( lv+1, amr->FluSg[lv+1], NULL_INT, NULL_INT, DATA_AFTER_REFINE, _TOTAL, _NONE, Flu_ParaBuf, USELB_YES );

      LB_Init_LoadBalance( Redistribute_Yes, SendGridData_Yes, Par_Weight, ResetLB_Yes, lv+1 );
#     endif

      if ( MPI_Rank == 0 )    Aux_Message( stdout, "done\n" );
   } // for (int lv=0; lv<MAX_LEVEL; lv++)

      if ( MPI_Rank == 0 )    Aux_Message( stdout, "Adding particles after restart ... done\n" );
   } // if ( amr->Par->Init == PAR_INIT_BY_RESTART  &&  OPT__RESTART_RESET  &&  Star_AddParForRestart )



// 2. set the central coordinates of tidal field
   if ( OPT__RECORD_USER  &&  Aux_Record_User_Ptr != NULL )
   {
      if ( MPI_Rank == 0 )    Aux_Message( stdout, "Setting initial central coordinates ... " );

      Aux_Record_User_Ptr();

      SetExtPotAuxArray_EridanusII( ExtPot_AuxArray_Flt, ExtPot_AuxArray_Int );
#     ifdef GPU
      CUAPI_SetConstMemory_ExtAccPot();
#     endif

      if ( MPI_Rank == 0 )    Aux_Message( stdout, "done\n" );
   }


   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ... done\n", __FUNCTION__ );

} // FUNCTION : Init_User_EridanusII



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
