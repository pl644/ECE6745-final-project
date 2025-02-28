//========================================================================
// ece6745-misc
//========================================================================
// _RISCV will only be defined when cross-compiling for RISCV.

#ifndef ECE6745_MISC_H
#define ECE6745_MISC_H

#ifndef _RISCV
#include <stdlib.h>
#endif

//------------------------------------------------------------------------
// ECE6745_UNUSED
//------------------------------------------------------------------------
// To avoid warnings/errors about unused variables.

#define ECE6745_UNUSED_INT( x )                                         \
  do {                                                                  \
    __attribute__((__unused__)) int unused_##x = (int) x;               \
  } while ( 0 )

#define ECE6745_UNUSED_PTR( x )                                         \
  do {                                                                  \
    __attribute__((__unused__)) void* unused_##x = (void*) x;           \
  } while ( 0 )

//------------------------------------------------------------------------
// ece6745_seed and ece6745_rand
//------------------------------------------------------------------------
// Sets seed and returns pseudo-random number

#ifdef _RISCV

void ece6745_srand( int seed );
int  ece6745_rand();

#else

inline
void ece6745_srand( unsigned int seed )
{
  srand( seed );
}

inline
int ece6745_rand()
{
  return (int) rand();
}

#endif

//------------------------------------------------------------------------
// ece6745_atoi
//------------------------------------------------------------------------
// Converts an string to an integer. This is kind of hacky, but for the
// cross-compiled version we assume the given ASCII character string has
// already been widened which is really only true when using this with
// argv.

#ifdef _RISCV

int ece6745_atoi( const char* str );

#else

inline
int ece6745_atoi( const char* str )
{
  return atoi(str);
}

#endif

//------------------------------------------------------------------------
// exit
//------------------------------------------------------------------------
// exit the program with the given status code

#ifdef _RISCV

inline
void ece6745_exit( int i )
{
  int msg = 0x00010000 | i;
  __asm__ ( "csrw 0x7c0, %0" :: "r"(msg) );
}

#else

inline
void ece6745_exit( int i )
{
  exit(i);
}

#endif

//------------------------------------------------------------------------
// ece6745_stats_on
//------------------------------------------------------------------------

#ifdef _RISCV

inline
void ece6745_stats_on()
{
  int status = 1;
  __asm__ ( "csrw 0x7c1, %0" :: "r"(status) );
}

#else

inline
void ece6745_stats_on()
{ }

#endif

//------------------------------------------------------------------------
// ece6745_stats_off
//------------------------------------------------------------------------

#ifdef _RISCV

inline
void ece6745_stats_off()
{
  int status = 0;
  __asm__ ( "csrw 0x7c1, %0" :: "r"(status) );
}

#else

inline
void ece6745_stats_off()
{ }

#endif

#endif /* ECE6745_MISC_H */

