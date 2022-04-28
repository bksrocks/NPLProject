from django.shortcuts import render


def default_page(request):
    return render(request=request,
                  template_name="default.html",
                  context={})

