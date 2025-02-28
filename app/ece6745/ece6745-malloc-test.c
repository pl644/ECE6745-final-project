//========================================================================
// Unit tests for malloc
//========================================================================

#include "ece6745-misc.h"
#include "ece6745-check.h"
#include "ece6745-wprint.h"
#include "ece6745-malloc.h"
#include "ece6745-bthread.h"

//------------------------------------------------------------------------
// Test malloc int
//------------------------------------------------------------------------

void test_case_1_malloc_int()
{
  ECE6745_CHECK( L"test_case_1_malloc_int" );

  int* a_ptr = ece6745_malloc( sizeof(int) );
  *a_ptr = 42;

  ECE6745_CHECK_INT_EQ( ece6745_get_heap_usage(), sizeof(int) );
  ECE6745_CHECK_INT_EQ( *a_ptr, 42  );

  ece6745_free(a_ptr);

  ECE6745_CHECK_INT_EQ( ece6745_get_heap_usage(), 0 );
}

//------------------------------------------------------------------------
// Test malloc struct
//------------------------------------------------------------------------

typedef struct
{
  int x;
  int y;
}
point_t;

void test_case_2_malloc_struct()
{
  ECE6745_CHECK( L"test_case_2_malloc_struct" );

  point_t* point_p = ece6745_malloc( sizeof(point_t) );
  point_p->x = -1;
  point_p->y =  1;

  ECE6745_CHECK_INT_EQ( ece6745_get_heap_usage(), sizeof(point_t) );
  ECE6745_CHECK_INT_EQ( point_p->x, -1 );
  ECE6745_CHECK_INT_EQ( point_p->y,  1 );

  ece6745_free(point_p);

  ECE6745_CHECK_INT_EQ( ece6745_get_heap_usage(), 0 );
}

//------------------------------------------------------------------------
// Test malloc array
//------------------------------------------------------------------------

void test_case_3_malloc_array()
{
  ECE6745_CHECK( L"test_case_3_malloc_array" );

  int* a = ece6745_malloc( 4*sizeof(int) );
  a[0] = 0;
  a[1] = 1;
  a[2] = 2;
  a[3] = 3;

  ECE6745_CHECK_INT_EQ( ece6745_get_heap_usage(), 4*sizeof(int) );
  ECE6745_CHECK_INT_EQ( a[0], 0 );
  ECE6745_CHECK_INT_EQ( a[1], 1 );
  ECE6745_CHECK_INT_EQ( a[2], 2 );
  ECE6745_CHECK_INT_EQ( a[3], 3 );

  ece6745_free(a);

  ECE6745_CHECK_INT_EQ( ece6745_get_heap_usage(), 0 );
}

//------------------------------------------------------------------------
// Test slist
//------------------------------------------------------------------------

typedef struct _slist_int_node_t
{
  int                       value;
  struct _slist_int_node_t* next_p;
}
slist_int_node_t;

typedef struct
{
  slist_int_node_t* head_p;
}
slist_int_t;

// Construct slist

void slist_int_construct( slist_int_t* this )
{
  this->head_p = NULL;
}

// Destruct slist

void slist_int_destruct( slist_int_t* this )
{
  while ( this->head_p != NULL ) {
    slist_int_node_t* temp_p
      = this->head_p->next_p;
    ece6745_free( this->head_p );
    this->head_p = temp_p;
  }
}

// Push value on front of slist

void slist_int_push_front( slist_int_t* this, int v )
{
  slist_int_node_t* new_node_p
    = ece6745_malloc( sizeof(slist_int_node_t) );
  new_node_p->value  = v;
  new_node_p->next_p = this->head_p;
  this->head_p       = new_node_p;
}

// Test function

void test_case_4_slist()
{
  ECE6745_CHECK( L"test_case_4_slist" );

  slist_int_t lst;
  slist_int_construct ( &lst );
  slist_int_push_front( &lst, 12 );
  slist_int_push_front( &lst, 11 );
  slist_int_push_front( &lst, 10 );

  ECE6745_CHECK_INT_EQ( ece6745_get_heap_usage(), 3*sizeof(slist_int_node_t) );
  ECE6745_CHECK_INT_EQ( lst.head_p->value,                 10 );
  ECE6745_CHECK_INT_EQ( lst.head_p->next_p->value,         11 );
  ECE6745_CHECK_INT_EQ( lst.head_p->next_p->next_p->value, 12 );

  slist_int_destruct  ( &lst );

  ECE6745_CHECK_INT_EQ( ece6745_get_heap_usage(), 0 );
}

//------------------------------------------------------------------------
// main
//------------------------------------------------------------------------

int main( int argc, char** argv )
{
  __n = ( argc == 1 ) ? 0 : ece6745_atoi( argv[1] );

  if ( (__n <= 0) || (__n == 1) ) test_case_1_malloc_int();
  if ( (__n <= 0) || (__n == 2) ) test_case_2_malloc_struct();
  if ( (__n <= 0) || (__n == 3) ) test_case_3_malloc_array();
  if ( (__n <= 0) || (__n == 4) ) test_case_4_slist();

  ece6745_wprintf( L"\n\n" );
  return ece6745_check_status;
}

