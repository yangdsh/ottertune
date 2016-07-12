'''
Created on Jul 11, 2016

@author: dvanaken
'''

def map_workload(X_client, y_client):
    from experiment import ExpContext
    
    if ExpContext.dbms.name == "mysql":
        return ("analysis_20160710-204911_exps_mysql_5.6_m3.xlarge_tpch_rr_sf10_tr1_t0_runlimited_w0-0-1-0-1-0-1-1-1-0-0-1-0-0-1-0-1-0-1-0-1-1_sl")
    elif ExpContext.dbms.name == "postgres":
        return ("analysis_20160710-204142_exps_postgres_9.3_m3.xlarge_tpch_rc_sf10_tr1_t0_runlimited_w0-0-1-0-1-0-1-1-1-0-0-1-0-0-1-0-1-0-1-0-1-1_sl")
    else:
        assert False
