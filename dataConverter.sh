#!/bin/sh
year="2013"

# cp ../../../hko/data/$year*.tar.gz .
tar xzf $year.tar.gz

for file in $(find ./$year -name 'RAD*.REF2256') # for file in ./$year/*  # ${myhome}/SPARNN/data/2015/*
do
    if [[ -f $file ]]; then
        # copy stuff ....
        # make_png.lin.x RAD130522024800.REF2256 256 2 1 RAD130522024800.REF2256-linear.png
        # make_png.lin.x f 256 2 1 RAD130522024800.REF2256-linear.png
	# echo $file
	IFS='/' read -a myarray <<< "$file"
	# echo "IP: ${myarray[1]}"
        output="${myarray[2]}-linear.png"
	echo $output
	/project/dygroup2/xiaohui/SPARNN/data/radar2PNGUtil/make_png.nobg.lin.x $file 256 2 1 $output 
    fi
done

python imageRenameCrop.py 13

