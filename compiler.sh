#!/bin/bash
# sudo su
# ./main.sh

rm -R accuracy.txt
rm -R filter_acc.txt
rm -R average.txt

for k in 5 6 7 8 9 10 11 12 13 14
        do
        for i in `seq 0 9`
                do
                for j in `seq 0 10`
                        do
                        python3 main.py $i $k > accuracy.txt
                        tail -n 1 accuracy.txt  >> filter_acc.txt
                        tail -n 1 accuracy.txt
                done

                sum=0
                index=0
                mean=0

                for j in $(cat filter_acc.txt)
                        do
                        index=$(bc -l <<<"${index}+1")
                        sum=$(bc -l <<<"${sum}+${j}")
                done

                mean=$(bc -l <<<"${sum}/${index}")
                echo "R:" $mean "- N. colunas:" $i "- N. pop:" $k

                rm -R accuracy.txt
                rm -R filter_acc.txt
        done
done