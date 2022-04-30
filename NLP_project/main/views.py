from django.shortcuts import render
from django.contrib.staticfiles.views import serve


def default_page(request):
    return render(request=request,
                  template_name="default.html",
                  context={})


# def model(request):
#     # return render(request=request,
#     #               template_name="NLP_project/main/static/main/model00/model.json",
#     #               content_type='application/json')
#     return serve(request, 'https://poem-maker.s3.amazonaws.com/model.json')
