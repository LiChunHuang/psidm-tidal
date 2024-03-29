#include "GAMER.h"

#if ( MODEL == ELBDM  &&  defined GRAVITY )

#include "CUPOT.h"

static real GetMaxPot( const int lv );


extern bool   Tidal_Enabled;
extern double Tidal_CutoffR;
extern double Tidal_CM[3];
extern int    Sponge_Mode;




//-------------------------------------------------------------------------------------------------------
// Function    :  ELBDM_GetTimeStep_Gravity
// Description :  Estimate the evolution time-step from the ELBDM potential energy solver
//
// Note        :  1. This function should be applied to both physical and comoving coordinates and always
//                   return the evolution time-step (dt) actually used in various solvers
//                   --> Physical coordinates : dt = physical time interval
//                       Comoving coordinates : dt = delta(scale_factor) / ( Hubble_parameter*scale_factor^3 )
//                   --> We convert dt back to the physical time interval, which equals "delta(scale_factor)"
//                       in the comoving coordinates, in Mis_GetTimeStep()
//                2. Time-step is set to restrict the real-space rotation angle to be "DT__GRAVITY*2*pi"
//                   --> dt = DT__GRAVITY*2*pi/Eta/Max(Pot)
//
// Parameter   :  lv : Target refinement level
//
// Return      :  dt
//-------------------------------------------------------------------------------------------------------
double ELBDM_GetTimeStep_Gravity( const int lv )
{

   real   MaxPot;
   double dt;

// get the maximum potential
   MaxPot = GetMaxPot( lv );

// get the time-step
   dt = DT__GRAVITY*2.0*M_PI/(ELBDM_ETA*MaxPot);

   return dt;

} // FUNCTION : ELBDM_GetTimeStep_Gravity



//-------------------------------------------------------------------------------------------------------
// Function    :  GetMaxPot
// Description :  Get the maximum potential at the target level among all MPI ranks
//
// Note        :  1. Invoked by ELBDM_GetTimeStep_Gravity()
//                2. Include gravitational and self-interaction potentials
//
// Parameter   :  lv : Target refinement level
//
// Return      :  MaxPot
//-------------------------------------------------------------------------------------------------------
real GetMaxPot( const int lv )
{
//   printf("CMx = %f, CMy = %f, CMz = %f, CutoffR = %f", Tidal_CM[0],Tidal_CM[1],Tidal_CM[2],Tidal_CutoffR);
   real   PotG, PotS;            // PotG/S: gravitational (both self- and external gravity) / self-interaction potential
   real   Pot, MaxPot=0.0;       // Pot = PotG + PotS
   double x0, y0, z0, x, y, z;
   int    SibPID;
   bool   Skip, AnyCell=false;
//   real   MinR = 10000;
const double Tidal_CutoffR2 = SQR( Tidal_CutoffR );
double dr[3], r2;


// get the maximum potential in this rank
#  pragma omp parallel for private( PotG, PotS, Pot, x0, y0, z0, x, y, z, SibPID, Skip, dr, r2 ) \
                           reduction( max:MaxPot ) reduction( ||:AnyCell ) schedule( runtime )
   for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
   {
//    if OPT__FIXUP_RESTRICT is enabled, skip all non-leaf patches not adjacent to any coarse-fine boundaries
//    because their data is not used for providing ghost boundaries and they are later overwritten by the refined patches
//    we still need to update their mass density for the Poisson solver but their phase is irrelevant
//    so they need not be considered in the gravity time step calculation
//    note that this leads to the gravity timestep being "inf" when a level is completely refined
      if ( OPT__FIXUP_RESTRICT ) {
         Skip = true;

         if ( amr->patch[0][lv][PID]->son == -1 )  Skip = false;
         else
         {
            for (int s=0; s<26; s++)
            {
               SibPID = amr->patch[0][lv][PID]->sibling[s];

//             proper-nesting check
#              ifdef GAMER_DEBUG
               if ( SibPID == -1 )  Aux_Error( ERROR_INFO, "SibPID == -1 (Lv %d, PID %d) !!\n", lv, PID );
#              endif

//             non-periodic BC.
               if ( SibPID < -1 )   continue;

//             check whether this patch is adjacent to a coarse-fine boundary
               if ( amr->patch[0][lv][SibPID]->son == -1 )
               {
                  Skip = false;
                  break;
               }
            } // for (int s=0; s<26; s++)
         } // if ( amr->patch[0][lv][PID]->son == -1 ) ... else ...
      } else { // if ( OPT__FIXUP_RESTRICT ) {
         Skip = false;
      } // if ( OPT__FIXUP_RESTRICT ) { ... else

      if ( Skip )    continue;
      else           AnyCell = true;


//    calculate the potential
      const double dh = amr->dh[lv];

      x0 = amr->patch[0][lv][PID]->EdgeL[0] + 0.5*dh;
      y0 = amr->patch[0][lv][PID]->EdgeL[1] + 0.5*dh;
      z0 = amr->patch[0][lv][PID]->EdgeL[2] + 0.5*dh;

      for (int k=0; k<PATCH_SIZE; k++) {  z = z0 + (double)k*dh;
      for (int j=0; j<PATCH_SIZE; j++) {  y = y0 + (double)j*dh;
      for (int i=0; i<PATCH_SIZE; i++) {  x = x0 + (double)i*dh;


// skip cells outside the cut-off radius
if ( Tidal_Enabled  &&  Sponge_Mode != 3 )
{
   dr[0] = x - Tidal_CM[0];
   dr[1] = y - Tidal_CM[1];
   dr[2] = z - Tidal_CM[2];
   r2    = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
   
//   MinR = MIN( MinR, SQRT(r2));
   
   if ( r2 > Tidal_CutoffR2 )    continue;
//   else printf("r2 = %f", r2);
}


         PotG   = amr->patch[ amr->PotSg[lv] ][lv][PID]->pot[k][j][i];
#        ifdef QUARTIC_SELF_INTERACTION
         PotS   = ELBDM_LAMBDA*amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[DENS][k][j][i];
#        else
         PotS   = (real)0.0;
#        endif

         Pot    = FABS( PotG + PotS );    // remember to take the absolute value
         MaxPot = MAX( MaxPot, Pot );
      }}} // k,j,i
   } // for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
//   printf("MinR = %f", MinR);

// get the maximum potential in all ranks
   real MaxPot_AllRank;
   bool AnyCell_AllRank;
#  ifdef FLOAT8
   MPI_Allreduce( &MaxPot, &MaxPot_AllRank, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
#  else
   MPI_Allreduce( &MaxPot, &MaxPot_AllRank, 1, MPI_FLOAT,  MPI_MAX, MPI_COMM_WORLD );
#  endif
   MPI_Reduce( &AnyCell, &AnyCell_AllRank, 1, MPI_C_BOOL, MPI_LOR, 0, MPI_COMM_WORLD );


// check
//   if ( MaxPot_AllRank == 0.0  &&  AnyCell  &&  MPI_Rank == 0 )
//      Aux_Error( ERROR_INFO, "MaxPot == 0.0 at lv %d !!\n", lv );


   return MaxPot_AllRank;

} // FUNCTION : GetMaxPot



#endif // #if ( MODEL == ELBDM  &&  defined GRAVITY )
