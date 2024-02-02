//Atomic add function to work with floats
inline void atomicAddFloat(volatile __global float *addr, float val)
{
	//Create a union object between uint and float
	union 
	{
		unsigned int u32;		//Unsigned integer
		float        f32;		//Float
	} next, expected, current;	//Define three variables based on the defined union
	//Set to addr
	current.f32 = *addr;
	//While current does not equal the expected do the following
	do 
	{
		//Set expected to current
		expected.f32 = current.f32;
		//Set next to the expected + val
		next.f32 = expected.f32 + val;
		//Perform atomic cmpxchg with addr, expected and next
		current.u32 = atomic_cmpxchg((volatile __global unsigned int *)addr,
			expected.u32, next.u32);
	} 
	while (current.u32 != expected.u32);
}

//Reduce add kernal
__kernel void reduce_add_4(__global const float* A, __global float* B, __local float* scratch) 
{
	//Get the global ID
	int id = get_global_id(0);
	//Get the local ID
	int lid = get_local_id(0);
	//Get the local size
	int N = get_local_size(0);
	//Cache all N values from global memory to local memory
	scratch[lid] = A[id];
	//Wait for all local threads to finish copying from global to local memory
	barrier(CLK_LOCAL_MEM_FENCE);
	//Loop through local memory
	for (int i = 1; i < N; i *= 2) 
	{
		//Check if to operate, if yes then add the current local value to the next
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];
		//Wait for all local threads to finish copying from global to local memory
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	//Add all of the values from local back to global position 0
	if (!lid) 
		atomicAddFloat(&B[0],scratch[lid]);
}

//Reduce standard deviation kernal
__kernel void reduce_STD_4(__global const float* A, __global float* B, __local float* scratch, float Mean, int Padd)
{
	//Get global ID
	int id = get_global_id(0);
	//Get local ID
	int lid = get_local_id(0);
	//Get local size
	int N = get_local_size(0);
	//Get global size
	int NG = get_global_size(0);
	//Set size to be the size minus the padding value
	int Size = NG-Padd;
	//Cache all N values from global memory to local memory, that are less than NG-Padd
	if (id < (NG- Padd))
	{
		scratch[lid] = A[id];
		//Perform first operation for standard deviation
		scratch[lid] = ((scratch[lid] - Mean) * (scratch[lid] - Mean));
	}
	//Wait for all local threads to finish copying from global to local memory
	barrier(CLK_LOCAL_MEM_FENCE);
	//Loop through all local memory
	for (int i = 1; i < N; i *= 2) 
	{
		//Check if to operate, if yes then add the current local value to the next
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] += scratch[lid + i];
		//Wait for all local threads to finish copying from global to local memory
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	//Add all of the values from local back to global position 0
	if (!lid) 
		atomicAddFloat(&B[0], scratch[lid]);
}

//Selection sort kernal
__kernel void ParallelSelection(__global const float* A, __global float* B)
{
	//Get global ID
	int i = get_global_id(0);
	//Get global size
	int n = get_global_size(0);
	//Set iKey to the current value
	float iKey = A[i];
	//Set pos to 0
	int pos = 0;
	//Loop through the size
	for (int j = 0; j<n; j++)
	{
		//Get current value at j
		float jKey = A[j];
		//Get the smaller out of the two values
		bool smaller = (jKey < iKey) || (jKey == iKey && j < i); 
		//Position + the result of smaller
		pos += (smaller) ? 1 : 0;
	}
	//Put variable into output
	B[pos] = A[i];
}

///////////////////////////////////////////////////////////////////////////////////////////
//MIN and Max functions not being used as it is more optimal to simply sort the vector and
//                        then get the first and last elements. 
///////////////////////////////////////////////////////////////////////////////////////////

//Reduce minimum kernal gets smallest value
__kernel void reduce_MIN(__global const int* A, __global int* B, __local float* scratch)
{
	//Get global ID
	int id = get_global_id(0);
	//Get local ID
	int lid = get_local_id(0);
	//Get local size
	int N = get_local_size(0);
	//Get global size
	int NG = get_global_size(0);
	//Cache all N values from global memory to local memory
	scratch[lid] = A[id];
	//Wait for all local threads to finish copying from global to local memory
	barrier(CLK_LOCAL_MEM_FENCE);
	//Loop through local memory
	for (int i = 1; i < N; i *= 2)
	{
		//Check if to operate, if yes then get the minimum between the current and next value
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] = min(scratch[lid], scratch[lid + i]);
		//Wait for all local threads to finish copying from global to local memory
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	//Fill output with the minimum value between the current value and the one found from this execution
	if (!lid)
		atomic_min(&B[0], scratch[lid]);
}

//Reduce maximum kernal gets biggest value
__kernel void reduce_MAX(__global const int* A, __global int* B, __local float* scratch)
{
	//Get global ID
	int id = get_global_id(0);
	//Get local ID
	int lid = get_local_id(0);
	//Get local size
	int N = get_local_size(0);
	//Get global size
	int NG = get_global_size(0);
	//Cache all N values from global memory to local memory
	scratch[lid] = A[id];
	//Wait for all local threads to finish copying from global to local memory
	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory
	//Loop through local memory
	for (int i = 1; i < N; i *= 2)
	{
		//Check if to operate, if yes then get the maximum between the current and next value
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] = max(scratch[lid], scratch[lid + i]);
		//Wait for all local threads to finish copying from global to local memory
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	//Fill output with the minimum value between the current value and the one found from this execution
	if (!lid)
		atomic_max(&B[0], scratch[lid]);
}