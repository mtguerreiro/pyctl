default: all

all: cdmpc_py

cdmpc_py: cdmpc_py.c
	gcc -fPIC mvops.c -c
	gcc -shared -o libmvops.so mvops.o
	
	gcc -fPIC qp.c -c
	gcc -shared -o libqp.so qp.o
	
	gcc -fPIC dmpc_matrices.c -c
	gcc -shared -o libdmpc_matrices.so dmpc_matrices.o
	
	gcc -fPIC dmpc.c -c
	gcc -L. -Wl,-rpath=. -shared -o libdmpc.so dmpc.o -L. -lmvops -lqp -ldmpc_matrices
	
	gcc -fPIC cdmpc_py.c -c
	gcc -L. -Wl,-rpath=. -shared -o libcdmpc_py.so cdmpc_py.o -L. -ldmpc

clean:
	$(RM) *.o
	$(RM) *.so
