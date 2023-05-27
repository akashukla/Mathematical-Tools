#!/bin/bash
export gComDir="/home/akash/g0g2/gkylsoft/gkyl/bin/"


rm allcell.csv
rm allcellad.csv

for n in 4 8 16 32
do
  x=$n
  y=$n
  z=$n
  $gComDir/gkyl -e "xcells=$x;ycells=$y;zcells=$z" error.lua 
done

python3 ploterror.py
