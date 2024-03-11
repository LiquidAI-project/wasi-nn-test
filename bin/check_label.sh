#!/bin/bash

index=$1;
label=$(cat models/labels.txt | grep "$index;" | cut -d';' -f2);

echo "Index ${index} corresponds to label: ${label}";
