# DPA Binary Compilation Guide for Linux / qBraid

This guide provides instructions for compiling the **Defining Point Algorithm (DPA)** C++ binary (`dpa-main`) on Linux environments (e.g., qBraid Jupyter Hub containers, Ubuntu/Debian Linux nodes, HPC clusters).

---

## 1. Overview & Architecture Difference

- **Pre-compiled macOS Binary:** The default `dpa-main` file located in `doe/phase_3/code/dpa-main` is compiled for macOS (`Mach-O 64-bit arm64`).
- **Linux Execution (qBraid):** Linux requires an ELF 64-bit binary (`ELF 64-bit LSB executable x86-64`). Running the macOS binary directly on qBraid will result in an `Exec format error` or `cannot execute binary file`.
- **Source Code Location:** The C++ source files, headers, and Makefile are located at:
  ```text
  doe/external/code/dpa/
  ├── main.cpp
  ├── Makefile
  ├── build.txt
  └── classes/
      ├── include/
      ├── src/
      └── obj/
  ```

---

## 2. Prerequisites for Linux / qBraid

Before compiling, ensure the following are installed in your Linux environment:

1. **C++ Compiler & Build Tools:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential g++ clang make
   ```

2. **IBM ILOG CPLEX Optimization Studio (C++ API):**
   - CPLEX Studio must be installed on the Linux system (e.g., at `/opt/ibm/ILOG/CPLEX_Studio2211` or in your user home directory `~/CPLEX_Studio2211`).
   - The installation must contain the Concert Technology C++ headers and static libraries:
     - `cplex/include/ilcplex/cplex.h`
     - `concert/include/ilconcert/ilomodel.h`
     - `cplex/lib/<SYSTEM>/<LIBFORMAT>/libcplex.a`
     - `concert/lib/<SYSTEM>/<LIBFORMAT>/libconcert.a`

---

## 3. Step-by-Step Compilation Instructions

### Step 1: Set the CPLEX Environment Variable
Export the path to your Linux CPLEX Studio installation:
```bash
export CPLEX_DIRECTORY=/opt/ibm/ILOG/CPLEX_Studio2211
```
*(Adjust the directory path above to match your actual CPLEX installation path on qBraid/Linux).*

### Step 2: Configure `Makefile` for Linux Architecture
Navigate to the DPA source directory:
```bash
cd doe/external/code/dpa
```

Open `Makefile` and update the platform-specific settings for Linux:

```makefile
# For Linux x86_64:
SYSTEM     = x86-64_linux
LIBFORMAT  = static_pic

# Compiler selection for Linux (g++ or clang++)
CCC        = g++ -O3

# CPLEX directory from environment
CPLEX_DIRECTORY ?= /opt/ibm/ILOG/CPLEX_Studio2211
CPLEXDIR        = $(CPLEX_DIRECTORY)/cplex
CONCERTDIR      = $(CPLEX_DIRECTORY)/concert

# Libraries and Includes
CPLEXLIBDIR   = $(CPLEXDIR)/lib/$(SYSTEM)/$(LIBFORMAT)
CONCERTLIBDIR = $(CONCERTDIR)/lib/$(SYSTEM)/$(LIBFORMAT)

CONCERTINCDIR = $(CONCERTDIR)/include
CPLEXINCDIR   = $(CPLEXDIR)/include

EXDIR         = ./classes
EXINC         = $(EXDIR)/include
EXOBJ         = $(EXDIR)/obj
EXSRC         = $(EXDIR)/src

CCOPT         = -Wall -fPIC -Ofast -fexceptions -DIL_STD
CCFLAGS       = $(CCOPT) -I$(CPLEXINCDIR) -I$(CONCERTINCDIR) -I$(EXINC)
CCLNFLAGS     = -L$(CPLEXLIBDIR) -lilocplex -lcplex -L$(CONCERTLIBDIR) -lconcert -lm -pthread

main: main.cpp DefiningPoint.o
	$(CCC) $(CCFLAGS) main.cpp $(EXOBJ)/DefiningPoint.o $(CCLNFLAGS) -o dpa-main

DefiningPoint.o:
	mkdir -p $(EXOBJ)
	$(CCC) -c $(CCFLAGS) $(EXSRC)/DefiningPoint.cpp -o $(EXOBJ)/DefiningPoint.o

clean:
	rm -rf $(EXOBJ) dpa-main
```

### Step 3: Run `make`
Compile the executable:
```bash
make clean
make
```

### Step 4: Verify the Linux Binary
Check that the binary was built successfully as a Linux ELF executable:
```bash
file dpa-main
```
*Expected Output:*
```text
dpa-main: ELF 64-bit LSB executable, x86-64, version 1 (SYSV), dynamically linked...
```

### Step 5: Copy the Linux Binary to Phase 3 Code Directory
Replace the pre-compiled macOS binary in Phase 3 with your newly built Linux binary:
```bash
cp dpa-main ../../../phase_3/code/dpa-main
chmod +x ../../../phase_3/code/dpa-main
```

---

## 4. Alternate Fallback (Pure Python CPLEX Baseline)

If CPLEX C++ Studio is not installed on the Linux environment:
- The main pipeline in `grid_opt.py` runs natively using `docplex` (IBM CPLEX Python API) and Qiskit.
- The C++ DPA binary is **only used for external classical benchmarking comparison** against exact MILP Pareto frontiers.
- The core **Benders-QAMOO** quantum-classical hybrid pipeline runs entirely in Python without requiring `dpa-main`.
