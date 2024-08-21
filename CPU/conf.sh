#!/bin/bash
SYSTEM="snellius_spack"

case $SYSTEM in
  aws-normal)
    source conf/aws-normal.sh
    ;;

  aws-hpc)
    source conf/aws-hpc.sh
    ;;

  gcp)
    source conf/gcp.sh
    ;;

  azure)
    source conf/azure.sh
    ;;

  oracle)
    source conf/oracle.sh
    ;;

  oracle-hpc)
    source conf/oracle-hpc.sh
    ;;

  daint)
    source conf/daint.sh
    ;;
  
  alps)
    source conf/alps.sh
    ;;

  deep-est)
    source conf/deep-est.sh
    ;;

  snellius)
    source conf/snellius.sh
    ;;

  snellius_spack)
    source conf/snellius_spack.sh
    ;;
  
  snellius-short-rome)
    source conf/snellius.sh
    ;;

  snellius-long-rome)
    source conf/snellius.sh
    ;;
  
  snellius-short-genoa)
    source conf/snellius.sh
    ;;

  snellius-long-genoa)
    source conf/snellius.sh
    ;;


  *)
    echo -n "Unknown SYSTEM "$SYSTEM
    ;;
esac
