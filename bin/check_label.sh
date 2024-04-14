#!/bin/bash

index=$1;

function echo_instructions {
   echo "Usage: ./check_label.sh <index>";
   echo "- <index> must be an integer between 1 and 1000 with no leading zeros";
}

if [[ ! "$index" =~ ^[1-9][0-9]*$ ]]
then
    echo_instructions;
elif (( "$index" < 1 || "$index" > 1000 ))
then
    echo_instructions;
else
    label=$(cat models/labels.txt | grep --max-count=1 "$index;" | cut -d';' -f2);
    echo "Index ${index} corresponds to label: ${label}";
fi
