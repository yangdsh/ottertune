from django.conf.urls import include, url
from django.contrib import admin
from django.contrib.staticfiles.views import serve
from django.views.decorators.cache import never_cache

from website import settings
from website import views as website_views

admin.autodiscover()

urlpatterns = [
    url(r'^signup/', website_views.signup_view, name='signup'),
    url(r'^login/', website_views.login_view, name='login'),
    url(r'^logout/$', website_views.logout_view, name='logout'),

    url(r'^ajax_new/', website_views.ajax_new),
#     url(r'^status/', website_views.ml_info),

    url(r'^new_result/', website_views.new_result),
#     url(r'^result/', website_views.result),
    url(r'^update_similar/', website_views.update_similar),

    url(r'^$', website_views.redirect_home),
    url(r'^projects/$', website_views.home, name='home'),

    url(r'^projects/(?P<project_id>[0-9]+)/edit/$', website_views.update_project, name='edit_project'),
    url(r'^projects/new/$', website_views.update_project, name='new_project'),
    url(r'^projects/(?P<project_id>[0-9]+)/apps$', website_views.project, name='project'),
    url(r'^projects/delete/$', website_views.delete_project, name="delete_project"),
    url(r'^projects/(?P<project_id>[0-9]+)/apps/(?P<app_id>[0-9]+)/$', website_views.application, name='application'),

    url(r'^projects/(?P<project_id>[0-9]+)/apps/delete/$', website_views.delete_application, name='delete_application'),
    url(r'^projects/(?P<project_id>[0-9]+)/apps/new/$', website_views.update_application, name='new_application'),
    url(r'^projects/(?P<project_id>[0-9]+)/apps/(?P<app_id>[0-9]+)/edit/$', website_views.update_application, name='edit_application'),
    
    url(r'^projects/(?P<project_id>[0-9]+)/apps/(?P<app_id>[0-9]+)/results/(?P<result_id>[0-9]+)/$', website_views.result, name='result'),
    url(r'^projects/(?P<project_id>[0-9]+)/apps/(?P<app_id>[0-9]+)/workloads/(?P<wkld_id>[0-9]+)/$', website_views.workload_info, name='workload'),
#     url(r'^projects/(?P<project_id>[0-9]+)/apps/(?P<app_id>[0-9]+)/workloads/(?P<wkld_id>[0-9]+)/edit/$', website_views.edit_workload, name='edit_wkld_conf'),

    url(r'^projects/(?P<project_id>[0-9]+)/apps/(?P<app_id>[0-9]+)/db_confs/(?P<dbconf_id>[0-9]+)/$', website_views.db_conf_view, name='db_confs'),
    url(r'^projects/(?P<project_id>[0-9]+)/apps/(?P<app_id>[0-9]+)/db_metrics/(?P<dbmet_id>[0-9]+)/$', website_views.db_metrics_view, name='db_metrics'),
    url(r'^ref/(?P<dbms_name>.+)/(?P<version>.+)/params/(?P<param_name>.+)/$', website_views.db_conf_ref, name="dbconf_ref"),
    url(r'^ref/(?P<dbms_name>.+)/(?P<version>.+)/metrics/(?P<metric_name>.+)/$', website_views.db_metrics_ref, name="dbmetrics_ref"),
    url(r'^projects/(?P<project_id>[0-9]+)/apps/(?P<app_id>[0-9]+)/results/(?P<result_id>[0-9]+)/status$', website_views.ml_info, name="status"),

    url(r'^get_workload_data/', website_views.get_workload_data),
    url(r'^get_data/', website_views.get_timeline_data),

    # Uncomment the admin/doc line below to enable admin documentation:
    # url(r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
    url(r'^admin/', admin.site.urls),

    url(r'^static/(?P<path>.*)$', never_cache(serve)), 

]


if settings.DEBUG:
    import debug_toolbar

    urlpatterns = [
        url(r'^__debug__/', include(debug_toolbar.urls)),
    ] + urlpatterns

