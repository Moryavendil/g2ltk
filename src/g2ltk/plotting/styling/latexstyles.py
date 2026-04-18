### latex

from . import styled


def latex_preamble(style=None):
    if style=='thesis' or style=='thezpresz':
        return LATEX_STYLE_THESIS
    title = styled('latex_preamble', style=style)
    style_specific_latex_preamble = get_latex_preamble(title)

    return fr"""%
    %
    {general_latex_preamble}
    %
    {paper_specific_latex_preamble}
    %
    {style_specific_latex_preamble}
    %
    %"""

def get_latex_preamble(title):
    if title=='jfm':
        return jfm_latex_preamble
    if title=='thesis':
        return thesis_latex_preamble
    return "%"


general_latex_preamble = r"""%
%
%%% QOL
%
\setlength{\parindent}{0pt}% no indent
%"""

jfm_latex_preamble = r"""%
%
%%% JFM
%
\usepackage{newtxtext}
\usepackage{newtxmath}
%"""


paper_specific_latex_preamble = r"""%
%
%%% MACROS
%
%%% NOTATIONS DE LA THÈSE

% Mecanique des fluides
\newcommand\Rey{\mbox{\textit{Re}}}  % Reynolds number
\newcommand\Pran{\mbox{\textit{Pr}}} % Prandtl number, cf TeX's \Pr product
\newcommand\Pen{\mbox{\textit{Pe}}}  % Peclet number
\newcommand\Ca{\mbox{\textit{Ca}}}  % Capillary number
\newcommand\Bon{\mbox{\textit{Bo}}}  % Capillary number

\newcommand{\lcap}{\ensuremath{\ell_\text{c}}} % Longueur capilalire l_c or l_cap or l_gamma

% Rivelet
\newcommand{\mucl}{{\ensuremath{\mu_\text{cl}}}}
\newcommand{\vcap}{{\ensuremath{v_\text{c}}}} % capillary speed v_c or v_cap or v_gamma
\newcommand{\hlim}{{\ensuremath{h_\infty}}}
\newcommand{\vdrift}{v_\text{drift}}

\newcommand{\adv}{{\ensuremath{\partial_a}}} % advection operator = \partial_t + u_0 \partial_x

% Standard math
\newcommand\bnabla{\boldsymbol{\nabla}}
\newcommand\bcdot{\boldsymbol{\cdot}}

\newcommand{\ux}{{\ensuremath{u}}}
\newcommand{\uz}{{\ensuremath{v}}}
\newcommand{\us}{{\ensuremath{u_s}}}
\newcommand{\un}{{\ensuremath{u_n}}}
%"""


thesis_latex_preamble = r"""%
%
%%% THESIS
%
%%% Fonts and characters
% Encoding
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
%
% Font : Erewhon (Utopia-inspired)
\usepackage{amsmath}			% Permet de taper des formules mathématiques
\usepackage{amssymb}			% Permet d'utiliser des symboles mathématiques
\usepackage{amsfonts}			% Permet d'utiliser des polices mathématiques
\usepackage[erewhon]{newtxmath} % font for math : utopia for bold greek. option: [frenchmath] makes everything ugly
\usepackage{erewhon} % font for text : simili-utopia
%
% Special symbols
\usepackage[french,english]{babel} % Languages
\usepackage{pifont} % dingbats
\usepackage{circledsteps} % faire des trucs en dedans de ronds
%"""


LATEX_STYLE_THESIS = r"""
%%% Fonts and characters
%
% Encoding
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
%
% Font : Erewhon (Utopia-inspired)
\usepackage{amsmath}			% Permet de taper des formules mathématiques
\usepackage{amssymb}			% Permet d'utiliser des symboles mathématiques
\usepackage{amsfonts}			% Permet d'utiliser des polices mathématiques
\usepackage[erewhon]{newtxmath} % font for math : utopia for bold greek. option: [frenchmath] makes everything ugly
\usepackage{erewhon} % font for text : simili-utopia
%
% Special symbols
\usepackage[french,english]{babel} % Languages
\usepackage{pifont} % dingbats
\usepackage{circledsteps} % faire des trucs en dedans de ronds
%
%%% Maths
%\usepackage{nicefrac}			% Fractions 'inline'
%\numberwithin{equation}{chapter}
%\numberwithin{figure}{chapter}
%
%%% Physics
%
\usepackage{siunitx} % dimensioned quantities
\sisetup{
	mode=text, % TODO TODO TODO IMPORTANT FIXME mettre mode=match
	per-mode = symbol,
	range-phrase = --,
	exponent-product = \cdot, % pour avoir 1.5e10 -> 1.5 \cdot 10^{15}
%    print-unity-mantissa=true,
%    product_units = repeat,
	inter-unit-product=\ensuremath{{}\cdot{}}, % pour avoir N \cdot m instead of N m
	separate-uncertainty=true, % pour avoir 25.8 \pm 0.2 plutôt que 25.8(2)
	range-units = single}
\DeclareSIUnit{\pixel}{px}
\DeclareSIUnit{\px}{px}
\DeclareSIUnit{\frame}{frame}
\DeclareSIUnit\litre{l} % redefine litre to have a small l
%
\usepackage{physics} % Physics-friendly commands


%
%
%
\usepackage{xcolor}
\setlength{\parindent}{0pt} % no indent
%
%
%


\newcommand{\anglais}[1]{\textsl{#1}}
\newcommand{\neolo}[1]{{\og #1\fg{}}}
\renewcommand{\emph}[1]{\textsl{#1}} % emphasis (lighter italic)


%%% NOTATIONS DE LA THÈSE

% Mecanique des fluides
\newcommand{\dimensionlessnumber}[1]{\mbox{\textit{#1}}} % option 1
%\newcommand{\dimensionlessnumber}[1]{{\ensuremath{\mathrm{#1}}}} % option 2
\newcommand\Rey{\dimensionlessnumber{Re}}  % Reynolds number
\newcommand\Pra{\dimensionlessnumber{Pr}} % Prandtl number, cf TeX's \Pr product
\newcommand\Pen{\dimensionlessnumber{Pe}}  % Peclet number
\newcommand\Ca{\dimensionlessnumber{Ca}}  % Capillary number
\newcommand\Bon{\dimensionlessnumber{Bo}}  % Bond number
\newcommand\Web{\dimensionlessnumber{We}}  % Weber number

\newcommand{\lcap}{\ensuremath{\ell_\text{c}}} % Longueur capilalire l_c or l_cap or l_gamma


% Rivelet
\newcommand{\mucl}{{\ensuremath{\mu_\text{cl}}}}
\newcommand{\upmucl}{{\ensuremath{\muup_\text{cl}}}}
\newcommand{\vcap}{{\ensuremath{v_\text{c}}}} % capillary speed v_c or v_cap or v_gamma
\newcommand{\hlim}{{\ensuremath{h_\infty}}}
\newcommand{\vdrift}{{\ensuremath{v_\text{drift}}}}
\newcommand{\mutot}{{\ensuremath{\mu_\text{tot}}}}
\newcommand{\upmutot}{{\ensuremath{\muup_\text{tot}}}}

%%% THIN FILMS
\newcommand{\hcrest}{{\ensuremath{h_\text{c}}}}
\newcommand{\fr}{{\ensuremath{\vb{q}}}}
\newcommand{\zc}{{\ensuremath{z_\text{c}}}}


\newcommand{\adv}{{\ensuremath{\partial_a}}} % advection operator = \partial_t + u_0 \partial_x

\newcommand{\vunit}[1]{{\ensuremath{\vb{\hat{#1}}}}} % unit vector

\newcommand{\adim}[1]{{\ensuremath{ \tilde{#1} }}} % adimensionné

% Standard math
\newcommand\bnabla{\boldsymbol{\nabla}}
\newcommand\bcdot{\boldsymbol{\cdot}}

\newcommand{\conjugate}[1]{{\ensuremath{ {#1}^* }}}
\newcommand{\conj}[1]{{\conjugate{#1}}}

\newcommand{\ux}{{\ensuremath{u}}}
\newcommand{\uz}{{\ensuremath{v}}}
\newcommand{\us}{{\ensuremath{u_t}}}
\newcommand{\ut}{{\ensuremath{u_t}}}
\newcommand{\un}{{\ensuremath{u_n}}}

\newcommand{\upk}{\mathrm{k}}
\newcommand{\upv}{\mathrm{v}}
\newcommand{\upB}{\mathrm{B}}

% epsilons
\newcommand{\eps}{\upepsilon} % small parameter
\newcommand{\levicivita}{\epsilon} % symbol of levi-civita

% Operateurs linéaires
\newcommand{\Lz}{{\ensuremath{\mathcal{L}_z}}}
\newcommand{\Lw}{{\ensuremath{\mathcal{L}_w}}}
% Opérateurs non-linéaires
\newcommand{\NLzz}{{\ensuremath{\mathcal{N}_{z}}}}
\newcommand{\NLww}{{\ensuremath{\mathcal{N}_{w}}}}

\newcommand{\Tdim}{{\ensuremath{ \tilde T }}}

\newcommand{\cc}{{\text{c.c.}}}
% Equal by definition. What to choose?
% * hat delta (ugly and weird)
% * hat equal (means estimator according to wikipedia)
% * := (brings a computer science vibe, not very mathy),
% * 3 horizontal bars (avoid, reserved for modulo operations)
%\newcommand*{\deq}{%
%% This garbage fire of a command is due to G. Bermudez.
%% Even himself can't remember what any part of this mess means,
%% so good luck to whomever has to debug it.
%\mathrel{\vbox{\offinterlineskip\ialign{%
%\hfil##\hfil\cr
%\scalebox{1.13}[1.06]{\textasciicircum}\cr
%\noalign{\kern-1.1ex}
%$=$\cr
%}}}}
\makeatletter
\newcommand*{\defeq}{\mathrel{\rlap{%
\raisebox{0.3ex}{$\m@th\cdot$}}%
\raisebox{-0.3ex}{$\m@th\cdot$}}%
=}
\newcommand*{\eqdef}{=\mathrel{\rlap{%
\raisebox{0.3ex}{$\m@th\cdot$}}%
\raisebox{-0.3ex}{$\m@th\cdot$}}%
}
\makeatother

%%% COULEURS
\definecolor{col_acoustic}{HTML}{0eb200}
\definecolor{col_inertia}{HTML}{09a9a9}
\definecolor{col_mucl}{HTML}{990000}
\definecolor{col_gammat}{HTML}{006699}
\definecolor{col_gamman}{HTML}{4d00e5}
\definecolor{col_geo}{HTML}{000000} %{777777}
\definecolor{col_nonres}{HTML}{444444}
%\definecolor{lightgray}{HTML}{444444}

\newcommand{\Cinertia}[1]{\textcolor{col_inertia}{#1}}
\newcommand{\Ccl}[1]{\textcolor{col_mucl}{#1}}
\newcommand{\Cgat}[1]{\textcolor{col_gammat}{#1}}
\newcommand{\Cgan}[1]{\textcolor{col_gamman}{#1}}
\newcommand{\Cgeo}[1]{\textcolor{col_geo}{#1}}
\newcommand{\Cac}[1]{\textcolor{col_acoustic}{#1}}
\newcommand{\Cnr}[1]{\textcolor{col_nonres}{#1}}

\newcommand{\Cgray}[1]{\textcolor{lightgray}{#1}}


%
%
%
%
%
%
"""

