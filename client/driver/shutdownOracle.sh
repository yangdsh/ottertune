#!/bin/sh
su - $oracle_sys <<EON
sqlplus / as sysdba <<EOF
shutdown immediate
exit
EOF
exit  
EON
