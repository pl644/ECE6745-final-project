#=========================================================================
# FC_Layer FL Model
#=========================================================================
'''
FIXED-POINT Q4.7 MULTIPLICATION REFERENCE GUIDE
===============================================

FORMAT OVERVIEW:
- Q4.7 = 12 bits total: 1 sign bit + 4 integer bits + 7 fractional bits
- Range: approximately -8.0 to +7.9375 with precision of 2^-7 (0.0078125)

MULTIPLICATION PROCESS:
1. When multiplying two Q4.7 numbers, the result expands to 24 bits:
   - 1 sign bit
   - 9 integer bits (4+4+1 for potential carry)
   - 14 fractional bits (7+7)

2. To convert back to Q4.7:
   a) Check for overflow (if result exceeds Q4.7 range)
   b) Shift right by 7 bits (equivalent to dividing by 2^7)
   c) Extract appropriate bits: sign bit + 4 integer bits + 7 fractional bits

OVERFLOW HANDLING:
- Overflow occurs if any bits beyond what Q4.7 can represent are significant
- Options: saturate to min/max value or signal overflow condition

EXAMPLE: Multiply 2.5 x (-1.75)
---------------------------------
Step 1: Represent in Q4.7
- 2.5  = 0 0010 1000000 (positive)
- -1.75 = 1 0001 1100000 (negative)

Step 2: Multiply and get 24-bit result
- 00101000000 x 00011100000 = 0000000100011100000000000
- Apply correct sign (pos x neg = neg): 1000000100011100000000000

Step 3: Shift right by 7 bits
- Before: 1000000100011100000000000
- After:  1000000000100011100...

Step 4: Extract result in Q4.7 format
- Result: 1 0100 0111000 = -4.375 in Q4.7
- Verify: 2.5 x (-1.75) = -4.375 ✓

KEY CONSIDERATIONS:
1. Always check for overflow by examining upper integer bits
2. Rounding can be applied during the conversion back to Q4.7
3. The fractional precision affects the accuracy of results
4. Negative numbers use two's complement representation
'''


from pymtl3 import *
from pymtl3.stdlib.stream.ifcs import IStreamIfc, OStreamIfc
from pymtl3.stdlib.stream import IStreamDeqAdapterFL, OStreamEnqAdapterFL
from pymtl3.passes.backends.verilog import *

# Define bit widths
Bits2 = mk_bits(2)
Bits4 = mk_bits(4)
Bits6 = mk_bits(6)
Bits8 = mk_bits(8)
Bits10 = mk_bits(10)
Bits12 = mk_bits(12)
Bits16 = mk_bits(16)
Bits18 = mk_bits(18)
Bits22 = mk_bits(22)
Bits24 = mk_bits(24)
Bits32 = mk_bits(32)

class FullyConnected_FL(Component):
  """
  Fully Connected Neural Network Layer (Functional Level Model)
  
  Message Structure (32-bit):
  
  For Weight Configuration (Type 0):
  [31:30] - Message Type (00)
  [29:20] - Input Channel Index (max 1024)
  [19:12] - Output Channel Index (max 256)
  [11:0]  - Weight Value (12-bit fixed-point q4.7)
  
  For Bias Configuration (Type 1):
  [31:30] - Message Type (01)
  [29:22] - Output Channel Index (max 256)
  [21:0]  - Bias Value (22-bit extended precision fixed-point)
  
  For Input Data (Type 2):
  [31:30] - Message Type (10)
  [29:26] - Batch Index (max 10)
  [25:16] - Input Channel Index (max 1024)
  [15:0]  - Input Value (16-bit q4.7 fixed-point with extra precision)
  
  For Output Data (Type 3):
  [31:30] - Message Type (11)
  [29:26] - Batch Index (max 10)
  [25:18] - Output Channel Index (max 256)
  [17:0]  - Output Value (18-bit q4.7 fixed-point with extra precision)
  """
  def construct(s, batch_size=4, input_channel=3, output_channel=2):
    # Interface
    s.istream = IStreamIfc(Bits32)  # Input stream
    s.ostream = OStreamIfc(Bits32)  # Output stream
    
    # Queue Adapters
    s.istream_q = IStreamDeqAdapterFL(Bits32)
    s.ostream_q = OStreamEnqAdapterFL(Bits32)
    
    s.istream //= s.istream_q.istream
    s.ostream //= s.ostream_q.ostream
    
    # Internal storage for weights, biases, and inputs
    s.weights = [[Wire(Bits12) for _ in range(output_channel)] for _ in range(input_channel)]
    s.biases = [Wire(Bits12) for _ in range(output_channel)]
    s.inputs = [[Wire(Bits12) for _ in range(input_channel)] for _ in range(batch_size)]
    s.input_counts = [Wire(Bits16) for _ in range(batch_size)]

    # Pending FIFO for output messages
    s.pending = []
    
    # FL block
    @update_once
    def block():
      tmp_product = Bits24(0)
      output_msgs = []

      # send one pending packet per cycle if any
      if s.pending and s.ostream_q.enq.rdy():
        s.ostream_q.enq(s.pending.pop(0))
        return

      if s.istream_q.deq.rdy() and s.ostream_q.enq.rdy():
        msg = s.istream_q.deq()
        
        # Extract message parts using bit slicing
        msg_type_bits = msg[30:32]
        
        # Process based on message type
        if msg_type_bits == 0:  # Weight config
          # Extract indices
          input_idx_bits = msg[20:30]
          output_idx_bits = msg[12:20]
          value_bits = msg[0:12]
          input_idx = input_idx_bits.uint()
          output_idx = output_idx_bits.uint()
          
          if input_idx < input_channel and output_idx < output_channel:
            s.weights[input_idx][output_idx] @= value_bits
        elif msg_type_bits == 1:  # Bias config
          # Extract data
          output_idx_bits = msg[22:30]
          value_bits = msg[0:12]
          
          output_idx = output_idx_bits.uint()
          
          if output_idx < output_channel:
            s.biases[output_idx] @= value_bits
        elif msg_type_bits == 2:  # Input data
          # Extract data
          batch_idx_bits = msg[26:30]
          channel_idx_bits = msg[16:26]
          value_bits = msg[0:12]
          
          batch_idx = batch_idx_bits.uint()
          channel_idx = channel_idx_bits.uint()
          
          if batch_idx < batch_size and channel_idx < input_channel:
            # Store input value
            s.inputs[batch_idx][channel_idx] @= value_bits
            # Update input counter using new_count for correct checking.
            tmp_count = s.input_counts[batch_idx]
            new_count = tmp_count + 1
            s.input_counts[batch_idx] @= new_count
            
            # Check if we've received all inputs for this batch
            if (tmp_count) == input_channel:
              # Process all outputs one by one
              for j in range(output_channel):
                # Start with bias
                accum = s.biases[j]
                # Do dot product with proper Q4.7 fixed-point arithmetic
                for k in range(input_channel):
                  input_extended = concat(Bits12(0), s.inputs[batch_idx][k])
                  weight_extended = concat(Bits12(0), s.weights[k][j])

                  tmp_product = input_extended * weight_extended

                  product_shifted = tmp_product[0:12]

                  accum = product_shifted + accum

                sign_bit = accum[11]
                sign_ext = Bits6(0) if sign_bit == 0 else Bits6(0x3F)
                result_value = concat(sign_ext, accum)

                type_bits = Bits2(3)
                batch_bits = Bits4(batch_idx)
                output_bits = Bits8(j)

                upper_part = concat(type_bits, batch_bits)
                header = concat(upper_part, output_bits)
                output_msg = concat(header, result_value)
                output_msgs.append(output_msg)

              # enqueue all generated outputs into pending FIFO
              s.pending.extend(output_msgs)
              # Reset counter
              s.input_counts[batch_idx] @= Bits16(0)
    
  # Line tracing
  def line_trace(s):
    return f"{s.istream}(){s.ostream}"
