#!/bin/bash
jupyter nbconvert --to webpdf --allow-chromium-download --execute --output-dir ./pdfs/ ./notebooks/*.ipynb 
exit
