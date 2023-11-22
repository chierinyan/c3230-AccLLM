tokreg="tok\/s: ([\.0-9]+)"
utreg="user: ([\.0-9]+)"
streg="system: ([\.0-9]+)"

function mark() {
    if [[ $1 =~ $tokreg ]]; then echo -n ${BASH_REMATCH[1]}; fi
    echo -n '|'
    if [[ $1 =~ $utreg ]]; then ut=${BASH_REMATCH[1]}; fi
    echo -n "${ut}|"
    if [[ $1 =~ $streg ]]; then st=${BASH_REMATCH[1]}; fi
    echo -n "${st}|"
    awk "BEGIN{print $ut/$st\"|\"}"
}

echo -n '|0|'
result=`./seq 233 | grep -e 'length:' -e 'main thread'`
mark "$result"

for i in 1 2 4 8 10 12 16; do
    echo -n "|${i}|"
    result=`./llama2 233 $i | grep -e 'length:' -e 'main thread'`
    mark "$result"
done
