LOGO FILES — INSTRUCTIONS
=========================

Place the following logo image files in this folder before compiling main.tex:

  epfl_logo.png
    Download from: https://www.epfl.ch/campus/services/communication/en/brand/logos/
    Recommended: "EPFL Logo" horizontal version, PNG with transparent background.
    Suggested height when placed: 2.2 cm (set in titlepage.tex).

  noa_logo.png
    Download from: https://www.noa.gr (or request from your NOA contact).
    Recommended: PNG with transparent background.
    Suggested height when placed: 2.2 cm (set in titlepage.tex).

If the logo files are not yet available, titlepage.tex contains commented-out
placeholder boxes that you can enable temporarily so the document still compiles.
To do so, comment out the \includegraphics lines and uncomment the \fbox lines
in titlepage.tex.

Supported formats: .png (preferred), .pdf, .jpg, .eps
Do NOT include the file extension in the \includegraphics command in titlepage.tex
(LaTeX will find the file automatically regardless of extension).
