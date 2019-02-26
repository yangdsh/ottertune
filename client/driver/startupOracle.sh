#!/bin/sh
su - $oracle_sys <<EON
sqlplus / as sysdba <<EOF
startup
quit
EOF
exit  
EON
