#CPLEX system and lib format
SYSTEM     = arm64_osx
LIBFORMAT  = static_pic

# CPLEX directory
CPLEX_DIRECTORY = /Users/nvenkat/Applications/CPLEX_Studio_Community2212
CPLEXDIR      = ${CPLEX_DIRECTORY}/cplex
CONCERTDIR    = ${CPLEX_DIRECTORY}/concert

# ---------------------------------------------------------------------
# Compiler selection 
# ---------------------------------------------------------------------

CCC = clang++ -O0

# ---------------------------------------------------------------------
# Compiler options 
# ---------------------------------------------------------------------

CCOPT = -Wall -fPIC -Ofast -fexceptions -DIL_STD -stdlib=libc++
#-O

#  ---------------------------------------------------------------------
# Link options and libraries
# ---------------------------------------------------------------------
#libraries
CPLEXLIBDIR   = $(CPLEXDIR)/lib/$(SYSTEM)/$(LIBFORMAT)
CONCERTLIBDIR = $(CONCERTDIR)/lib/$(SYSTEM)/$(LIBFORMAT)

#includes
CONCERTINCDIR = $(CONCERTDIR)/include
CPLEXINCDIR   = $(CPLEXDIR)/include

# ---------------------------------------------------------------------
# Source and include file directory
# ---------------------------------------------------------------------

EXDIR= ./classes


EXINC         = $(EXDIR)/include
EXOBJ	       = $(EXDIR)/obj
EXSRC         = $(EXDIR)/src

# ---------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------
#include flags
CCFLAGS = $(CCOPT) -I$(CPLEXINCDIR) -I$(CONCERTINCDIR) -I$(EXINC) -v


#library flags
CCLNFLAGS = -L$(CPLEXLIBDIR) -lilocplex -lcplex -L$(CONCERTLIBDIR) -lconcert -lm -pthread

$(info $$CPLEX_DIRECTORY is [${CPLEX_DIRECTORY}])
$(info $$CPLEXDIR is [${CPLEXDIR}])
$(info $$CPLEXINCDIR is [${CPLEXINCDIR}])
$(info $$CPLEXLIBDIR is [${CPLEXLIBDIR}])

main: main.cpp DefiningPoint.o
	${CCC} ${CCFLAGS} main.cpp $(EXOBJ)/DefiningPoint.o ${CCLNFLAGS} -o dpa-main 

DefiningPoint.o:
	mkdir -p $(EXOBJ)
	$(CCC) -c $(CCFLAGS) $(EXSRC)/DefiningPoint.cpp -o $(EXOBJ)/DefiningPoint.o




