//========================================================================
// ece6745-misc
//========================================================================

#include "ece6745-misc.h"

//------------------------------------------------------------------------
// ece6745_seed
//------------------------------------------------------------------------

#ifdef _RISCV

int ece6745_lfsr = 0x0000cafe;

void ece6745_srand( int seed )
{
  ece6745_lfsr = 0x0000ffff & seed;
}

#endif

//------------------------------------------------------------------------
// ece6745_rand
//------------------------------------------------------------------------
// Adapted from 16-bit LFSR from wikipedia:
//
//  https://en.wikipedia.org/wiki/Linear-feedback_shift_register
//

#ifdef _RISCV

int ece6745_rand()
{
  int x = ece6745_lfsr;
  int bit = (x ^ (x >> 1) ^ (x >> 3) ^ (x >> 12)) & 0x00000001;
  ece6745_lfsr = (ece6745_lfsr >> 1) | (bit << 15);

  return ece6745_lfsr;
}

#endif

//------------------------------------------------------------------------
// ece6745_atoi
//------------------------------------------------------------------------

#ifdef _RISCV

#include <wchar.h>

int ece6745_atoi( const char* str )
{
  // Hack! Only works because the argv strings have already been widened
  // by the simulator test harness.

  const wchar_t* wstr = (const wchar_t*) str;

  // Work from most- to least-significant digit. result*10 takes into
  // account the place value, and - L'0' subtracts the code for the
  // lowest digit to convert to an integer from 0 to 9.

  int result = 0;
  int i = 0;
  while ( wstr[i] != L'\0' ) {
    result = result*10 + wstr[i] - L'0';
    i += 1;
  }

  return result;
}

#endif

//------------------------------------------------------------------------
// dummy function
//------------------------------------------------------------------------

#ifndef _RISCV

// We always need at least something in an object file, otherwise
// the native build won't work. So we create a dummy function for native
// builds.

int ece6745_( int arg )
{
  return arg;
}

#endif

