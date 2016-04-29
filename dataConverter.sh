#!/bin/sh


for year in "2009" "2010" "2011" "2012" "2013" "2014"
do
    rm -r ./${year}
    # cp ../../../hko/data/$year*.tar.gz .

    tar xzf ${year}.tar.gz
    

    find ./${year} -type f -name 'RAD*.REF2256' -print0 | xargs -0 mv -t ./${year}
    find ./${year} type d -empty -delete

    for file in $(find ./$year -name 'RAD*.REF2256') # for file in ./$year/*  # ${myhome}/SPARNN/data/2015/*
    do
        if [[ -f $file ]]; then
            # copy stuff ....
            # make_png.lin.x RAD130522024800.REF2256 256 2 1 RAD130522024800.REF2256-linear.png
            # make_png.lin.x f 256 2 1 RAD130522024800.REF2256-linear.png
        # echo $file
        IFS='/' read -a myarray <<< "$file"
        echo "IP: ${myarray[2]}"
        echo ${file}
            output="${year}/${myarray[2]}-linear.png"
        echo $output
        /project/dygroup2/xiaohui/SPARNN/data/radar2PNGUtil/make_png.nobg.lin.x $file 256 2 1 $output
        fi
    done

python imageRenameCrop.py ${year}

done