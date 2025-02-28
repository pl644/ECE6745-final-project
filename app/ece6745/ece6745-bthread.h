//========================================================================
// ece6745-bthread
//========================================================================
// We use a very simple "runtime" to parallelize our apps. We called it
// bthread because it is a set of bare minimum "threading" library
// functions to parallelize a program across multiple cores.
//
// crt0.S has code that will make sure all cores call
// ece6745_bthread_init before main. Cores 1,2,3 stay in the
// ece6745_bthread_ini function, waiting in a loop we call the worker
// loop. Core 0 will return from ece6745_bthread_init and call main. In
// each iteration of the worker loop, a core will check if a flag is set
// by core 0. If it is, then it will execute the function core 0 stored
// in a shared location, then reset its flag. Cores other than core 0
// will stay in the worker loop indefinitely.
//
// We call core 0 the master core, and we call the other cores the worker
// cores. The master core "spawns work" on a worker core using the
// ece6745_bthread_spawn function. The master core needs the function
// pointer, the argument pointer, and a worker core's ID. It stores the
// function pointer and the argument pointer to a global location, then
// sets the flag for the given worker core.
//
// The master core can wait for a worker core to finish by using the
// ece6745_bthread_join function. It waits for a designated worker core
// until the worker core finishes executing its function (if any) then
// returns.

#ifndef ECE6745_BTHREAD_H
#define ECE6745_BTHREAD_H

//------------------------------------------------------------------------
// ece6745_bthread_get_num_cores
//------------------------------------------------------------------------
// Returns the number of cores.

#ifdef _RISCV

inline
int ece6745_bthread_get_num_cores()
{
  int num_cores;
  __asm__ ( "csrr %0, 0xfc1" : "=r"(num_cores) );
  return num_cores;
}

#else

inline
int ece6745_bthread_get_num_cores()
{
  return 1;
}

#endif

//------------------------------------------------------------------------
// ece6745_bthread_get_core_id
//------------------------------------------------------------------------
// Returns the core ID.

#ifdef _RISCV

inline
int ece6745_bthread_get_core_id()
{
  int core_id;
  __asm__ ( "csrr %0, 0xf14" : "=r"(core_id) );
  return core_id;
}

#else

inline
int ece6745_bthread_get_core_id()
{
  return 0;
}

#endif

//------------------------------------------------------------------------
// ece6745_bthread_set_num_workers/get_num_workers
//------------------------------------------------------------------------
// Not all cores need to be workers. We can only use one core even in a
// multicore system.

void ece6745_bthread_set_num_workers( int num_workers );
int  ece6745_bthread_get_num_workers();

//------------------------------------------------------------------------
// ece6745_bthread_init
//------------------------------------------------------------------------
// This function _MUST_ be called right at the beginning of the main().
// It will only let core 0 pass through. All other cores will be trapped
// in a worker loop, waiting be woken up by the core 0 (bthread_spawn).

void ece6745_bthread_init();

//------------------------------------------------------------------------
// bthread_spawn
//------------------------------------------------------------------------
// Spawn a function to a given worker core (thread). Need to provide:
//
//   thread_id     : ID of the thread we are spawning to
//   start_routine : Spawned function
//   arg           : A pointer to the argument.
//

typedef void (*ece6745_bthread_spawn_func)(void*);

int ece6745_bthread_spawn( int thread_id,
                           ece6745_bthread_spawn_func func,
                           void* arg );

//------------------------------------------------------------------------
// ece6745_bthread_join
//------------------------------------------------------------------------
// Wait for the given thread to finish executing its work.

int ece6745_bthread_join( int thread_id );

#endif /* ECE6745_BTHREAD_H */

