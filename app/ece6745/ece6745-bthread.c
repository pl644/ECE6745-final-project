//========================================================================
// ece6745-bthread
//========================================================================

#include "ece6745-bthread.h"
#include "ece6745-misc.h"

//------------------------------------------------------------------------
// Global data structures
//------------------------------------------------------------------------

// We need to initialize num workers to 1, because if ncores == 1 then
// bthread_init is never called.

int ece6745_bthread_num_workers = 1;

#ifdef _RISCV

ece6745_bthread_spawn_func ece6745_bthread_spawn_func_ptrs[4];
void* ece6745_bthread_spawn_func_args[4];
volatile int ece6745_bthread_flags[4] = {0,0,0,0};

#endif

//------------------------------------------------------------------------
// ece6745_bthread_init
//------------------------------------------------------------------------

#ifdef _RISCV

void ece6745_bthread_init()
{
  int core_id = ece6745_bthread_get_core_id();

  // If it's core zero, fall through

  if ( core_id == 0 ) {

    // Initialize num workers to be num cores

    int ncores = ece6745_bthread_get_num_cores();
    ece6745_bthread_num_workers = ncores;

    // Initialize flags to zero

    for ( int i = 0; i < 4; i++ )
      ece6745_bthread_flags[i] = 0;

    // More core zero to be busy

    ece6745_bthread_flags[0] = 1;

    return;
  }

  else {

    // Core 1-3 will wait here in the worker loop

    while (1) {

      // Wait until woken up by core 0. We insert some extra nops here to
      // avoid hammering on the memory system too hard.

      while( ece6745_bthread_flags[core_id] == 0 ) {
        __asm__ __volatile__ ( "nop;"
                               "nop;"
                               "nop;"
                               "nop;"
                               "nop;"
                               "nop;"
                               "nop;"
                               "nop;"
                               "nop;"
                               "nop;": : : "memory" );
      }

      // Execute the spawn function

      (*ece6745_bthread_spawn_func_ptrs[core_id])
        ( ece6745_bthread_spawn_func_args[core_id] );

      // Unset the flag so the master core knows this work core is done.

      ece6745_bthread_flags[core_id] = 0;

    }

  }
}

#else

void ece6745_bthread_init()
{
}

#endif

//------------------------------------------------------------------------
// ece6745_bthread_spawn
//------------------------------------------------------------------------
// Spawn a function to a given worker core (thread). Need to provide:
//
//   thread_id     : ID of the thread we are spawning to
//   start_routine : Spawned function
//   arg           : A pointer to the argument.
//

#ifdef _RISCV

int ece6745_bthread_spawn( int thread_id,
                           ece6745_bthread_spawn_func func,
                           void* arg )
{
  int nworkers = ece6745_bthread_get_num_workers();

  // If running with one worker, just call the function

  if ( nworkers == 1 ) {
    func( arg );
    return 0;
  }

  // Thread id out of range, return an error.

  if ( thread_id < 1 || thread_id >= 4 )
    return -1;

  // Check to see if the thread is already in use. If so, then return
  // an error.

  if ( ece6745_bthread_flags[thread_id] )
    return -1;

  // Set function and argument pointer

  ece6745_bthread_spawn_func_args[thread_id] = arg;
  ece6745_bthread_spawn_func_ptrs[thread_id] = func;

  // Wake up worker thread

  ece6745_bthread_flags[thread_id] = 1;

  return 0;
}

#else

int ece6745_bthread_spawn( int thread_id,
                           ece6745_bthread_spawn_func func,
                           void* arg )
{
  ECE6745_UNUSED_INT( thread_id );
  func( arg );
  return 0;
}

#endif

//------------------------------------------------------------------------
// ece6745_bthread_join
//------------------------------------------------------------------------
// Wait for the given thread to finish executing its work.

#ifdef _RISCV

int ece6745_bthread_join( int thread_id )
{
  int nworkers = ece6745_bthread_get_num_workers();

  // If running on a single-core system, just return

  if ( nworkers == 1 )
    return 0;

  // Thread id out of range, return an error.

  if( thread_id < 1 || thread_id >= 4 )
    return -1;

  // Wait until the given thread is no longer in use.

  while ( ece6745_bthread_flags[thread_id] ) {
    __asm__ __volatile__ ( "nop;"
                           "nop;"
                           "nop;"
                           "nop;"
                           "nop;"
                           "nop;"
                           "nop;"
                           "nop;"
                           "nop;"
                           "nop;": : : "memory" );
  }

  return 0;
}

#else

int ece6745_bthread_join( int thread_id )
{
  ECE6745_UNUSED_INT( thread_id );
  return 0;
}

#endif

//------------------------------------------------------------------------
// ece6745_bthread_set_num_workers/get_num_workers
//------------------------------------------------------------------------

#include "ece6745-wprint.h"

void ece6745_bthread_set_num_workers( int num_workers )
{
  int ncores = ece6745_bthread_get_num_cores();

  if ( num_workers > ncores )
    ece6745_bthread_num_workers = ncores;
  else
    ece6745_bthread_num_workers = num_workers;
}

int ece6745_bthread_get_num_workers()
{
  return ece6745_bthread_num_workers;
}

