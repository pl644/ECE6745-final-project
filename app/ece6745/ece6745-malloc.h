//========================================================================
// ece6745-malloc
// ========================================================================
// Very, very simple bump pointer malloc. Very easy to run out of memory.
// If you try and malloc too many bytes or the heap reaches the stack it
// will print out an error and exit the program. This malloc is _not_
// thread safe. If you call malloc on cores 1-3 it will cause an error.
// You can use ece6745_get_heap_usage to check if there are any memory
// leaks.

#ifndef ECE6745_MALLOC_H
#define ECE6745_MALLOC_H

#include "ece6745-misc.h"

#ifndef _RISCV
#include <stdlib.h>
#endif

//------------------------------------------------------------------------
// ece6745_malloc
//------------------------------------------------------------------------

void* ece6745_malloc( int mem_size );

//------------------------------------------------------------------------
// ece6745_free
//------------------------------------------------------------------------

void ece6745_free( void* ptr );

//------------------------------------------------------------------------
// ece6745_get_heap_usage
//------------------------------------------------------------------------

int ece6745_get_heap_usage();

#endif /* ECE6745_MISC_H */

