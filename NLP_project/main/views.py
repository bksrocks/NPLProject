from django.shortcuts import render
from django.contrib.staticfiles.views import serve


def default_page(request):
    return render(request=request,
                  template_name="default.html",
                  context={})


