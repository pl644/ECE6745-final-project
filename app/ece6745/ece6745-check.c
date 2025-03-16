//========================================================================
// ece2400-check.c
//========================================================================

#include "ece6745-check.h"

//------------------------------------------------------------------------
// Global Variables
//------------------------------------------------------------------------

int __n                  = 0;
int ece6745_check_status = 0;
int ece6745_check_expr0  = 0;
int ece6745_check_expr1  = 0;
float ece6745_check_float_expr0;
float ece6745_check_float_expr1;
float ece6745_check_float_diff;
float ece6745_check_float_epsilon;

