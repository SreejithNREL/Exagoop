find . -mindepth 1 ! -name 'cmake.sh' ! -name 'clean_build_files.sh' ! -name build_udf.sh  -exec rm -rf {} +

