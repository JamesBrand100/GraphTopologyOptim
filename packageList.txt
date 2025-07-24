\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{PurdueThesis}[2025-02-03 Purdue University thesis document class]
\newcommand{\PurdueThesisVersion}{PurdueThesis 2025-02-03}

%
%   [CMOS]
%     The Chicago Manual of  Style Online,
%     17th edition.
%     https://www.chicagomanualofstyle.org/home.html,
%     retrieved on 2020-02-29.
%
%   [WEB9]
%     Webster's Ninth New Collegiate Dictionary.
%

\LoadClass[12pt]{report}[2019/12/20]
\renewcommand{\contentsname}{TABLE OF CONTENTS}
\newlength{\ZZlengtha}

% Defines \AlphAlph so we can have over 26 appendices, chapter appendices, etc.
\usepackage{alphalph}

% Extend the array and tabular environments.
% See
%     https://www.ctan.org/pkg/array
\usepackage{array}

% Define
%     toprule[<width>]
%     midrule[<width>]
%     bottomrule[<width>]
% for tables.
% See
%     https://ctan.org/pkg/booktabs?lang=en
\usepackage{booktabs}

% Provide fancy boxes.
% See
%     https://ctan.org/pkg/fancybox?lang=en
\usepackage{fancybox}
  \newenvironment{CenteredBox}
    {\begin{Sbox}}%
    {%
      \end{Sbox}%
      \centerline{\parbox{\wd\@Sbox}{\TheSbox}}%
    }


% Provide color extensions.
% See
%     https://ctan.org/pkg/xcolor?lang=en
% This \usepackage is not in alphabetical order
% because the \definecolor below uses it.
\usepackage{xcolor}

% Use hyperlinks for cross-references.
%
% Based on information in
% http://ftp.math.purdue.edu/mirrors/ctan.org/macros/latex/contrib/hyperref/doc/manual.pdf
%     Make sure [hyperref -mark] comes last of your loaded packages,
%     to give it a fighting chance of not being over-written, since
%     its job is to redefine many LATEX commands.
%
% From
% http://mirrors.ibiblio.org/CTAN/macros/latex/contrib/biblatex/doc/biblatex.pdf
%     When using the hyperref package, it is preferable to load it
%     after biblatex.
%
% This must be loaded before the breakurl package.
%
% The options and what they do:
%     OPTION            DESCRIPTION
%     breaklinks        (default false) Allow links to break across lines by
%                       making links over multiple lines into PDF links to
%                       the same target.
%     colorlinks        (default false) Colors the text of links and anchors
%                       instead of putting ugly colored boxes around them.
%                       See anchorcolor, citecolor, filecolor, linkcolor,
%                       menucolor, and runcolor for color of links.
%     hypertexnames     (default true) Use guessable names for links.  The
%                       hypertexnames=false option makes sure that the link
%                       to the, for example, second appendix doesn't go to
%                       the second chapter.
%     linktoc           (default section) Make text (section), page number
%                       (page), both (all), or nothing (none) be link on
%                        TOC, LOF, and LOT.
%     ocgcolorlinks      Use colored links when viewed, but don't print
%                        links in a separate color.  A better implementation
%                        which hasn't the disadvantage to prevent line breaks
%                        is in the ocgx2 package.
%     pdfencoding        What kind of encoding to use for bookmarks and
%                        information directory entries.  'auto' uses
%                        PDFDocEncoding or, if necessary, unicode.  'pdfdoc'
%                        uses PDFDocEncoding.  'unicode' uses Unicode.
%     psdextra           Define more short names for PDF string commands.
%
%   I used 'Sky blue' for every link color as ok'd by Ashlee Messersmith
% <amiley@purdue.edu> and Anita Sale <aasale@iupui.edu>.  See Figure 2 of
%     https://www.nature.com/articles/nmeth.1618
% Sky blue in RGB is (86/255, 180/255, 233/255) = (0.337, 0.706, 0.914).
%   Sky blue was too light a color.  I've tried, from that same web page,
% 'Blue'.  'Blue' was still a little light.
%   Ashlee Messersmith suggested (0.129, 0.298, 0.808).
%   Ashlee Messersmith <amiley@purdue.edu> and Anita Sale <aasale@iupui.edu>
% both approved (0.129, 0.298, 0.808) after seeing it.
%
%   From section 3.5, "Extension Options", of
%     http://ctan.math.washington.edu/tex-archive/macros/latex/contrib/hyperref/doc/manual.pdf
%     OPTION       TYPE   COLOR      DESCRIPTION
%     linkcolor    color  red        Color for normal internal links.
%     anchorcolor  color  black      Color for anchor text.
%                                    Ignored by most drivers.
%     citecolor    color  green      Color for bibliographical citations
%                                    in text.
%     filecolor    color  cyan       Color for URLs which open local files.
%     menucolor    color  red        Color for Acrobat menu items.
%     runcolor     color  filecolor  Color for run links (launch annotations).
%     urlcolor     color  magenta    Color for linked URLs.
%     allcolors    color             Set all color options (without border and
%                                    field options).
%     Note that all color names must be defined before use, following the
%     normal system of the standard LaTeX color package.
%
%   I'm using the color name myblue to avoid any conflicts with any
% built-in LaTeX colors.
\definecolor{myblue}{rgb}{0.129, 0.298, 0.808}
\usepackage
  [
    allcolors     = myblue,
    breaklinks    = true,
    colorlinks    = true,
    hypertexnames = false,
    linktoc       = page,    % was 'section'
    pdfencoding   = auto,
    psdextra      = true,
  ]
  {hyperref}
    % Put URLs in roman font (instead of typewriter font) to save space.
    \urlstyle{rm}

% Used to create hyperlinks from a footnote to the occurence of the
% footnote in the main text.  This must come after \usepackage{hyperref}.
% See
%     https://ctan.org/pkg/footnotebackref
\usepackage[symbol=$\uparrow$]{footnotebackref}

% Provides the \ifpdf command.
% See
%     https://ctan.org/pkg/ifpdf?lang=en
\usepackage{ifpdf}

% The ifthen package gets loaded by something else and defines the
%     \ifthenelse
%     \newboolean
%     \provideboolean
%     \setboolean
%     \whiledo
% commands.
% See
%     https://ctan.org/pkg/ifthen?lang=en
% This defines
%     \ifthen{condition}{do if true}
\newcommand{\ifthen}[2]{\ifthenelse{#1}{#2}{}}

% From
%     https://tex.stackexchange.com/questions/147194/is-it-still-useful-to-load-the-lmodern-package/147206#147206
% I understand it is still a good idea to load lmodern.
\usepackage{lmodern}

% Place selected parts of a document in landscape mode.
% See
%     https://ctan.org/pkg/lscape?lang=en
\usepackage{lscape}

% Ltxtable provides the functionality of longtable and tabularx
% together inside a longtable environment.
% See
%     https://ctan.org/pkg/ltxtable?lang=en
\usepackage{ltxtable}

% The ntheorem package is used instead of amsthm so the counter
% a theorem-like construct uses can be overridden by the user.  See pages 4--5 of
%      http://tug.ctan.org/tex-archive/macros/latex/contrib/ntheorem/ntheorem.pdf
\usepackage[amsthm]{ntheorem}
  % From example 31 of https://math.mit.edu/~poonen/papers/writing.pdf
  %     Use a single numbering system for all theorems, lemmas, etc.,
  %     instead of having both a Theorem 1.1 and a Lemma 1.1 in the same
  %     paper. This makes statements easier to find...
  % For example,
  %     \newtheorem{lemma}[theorem]{Lemma}
  % has lemma use the theorem counter.
  \theoremstyle{plain}
  \newtheorem{theorem}{Theorem}[section]
  \newtheorem{lemma}[theorem]{Lemma}
  \newtheorem{corollary}[theorem]{Corollary}
  \newtheorem{proposition}[theorem]{Proposition}
  \newtheorem{conjecture}[theorem]{Conjecture}
  \newtheorem{criterion}[theorem]{Criterion}
  \newtheorem{assertion}[theorem]{Assertion}
  \theoremstyle{definition}
  \newtheorem{definition}[theorem]{Definition}
  \newtheorem{condition}[theorem]{Condition}
  \newtheorem{problem}[theorem]{Problem}
  \newtheorem{example}[theorem]{Example}
  \newtheorem{exercise}[theorem]{Exercise}
% \newtheorem{algorithm}[theorem]{Algorithm}
  \newtheorem{question}[theorem]{Question}
  \newtheorem{axiom}[theorem]{Axiom}
  \newtheorem{property}[theorem]{Property}
  \newtheorem{assumption}[theorem]{Assumption}
  \newtheorem{hypothesis}[theorem]{Hypothesis}
  \theoremstyle{remark}
  \newtheorem{remark}[theorem]{Remark}
  \newtheorem{note}[theorem]{Note}
  \newtheorem{notation}[theorem]{Notation}
  \newtheorem{claim}[theorem]{Claim}
  \newtheorem{summary}[theorem]{Summary}
  \newtheorem{acknowledgment}[theorem]{Acknowledgment}
  \newtheorem{case}[theorem]{Case}
  \newtheorem{conclusion}[theorem]{Conclusion}
  % I added "Observation" to the list of theorem-like constructs.
  \newtheorem{observation}[theorem]{Observation}

% Let links wrap around line and page breaks.
% See
%     https://ctan.org/pkg/ocgx2?lang=en
\usepackage[ocgcolorlinks]{ocgx2}

\usepackage{pgffor}

% Provide different sorts of rotation.
% See
%     https://ctan.org/pkg/rotating?lang=en
\usepackage{rotating}

% Provides support for setting the spacing between lines in a docment.
% See
%     https://ctan.org/pkg/setspace?lang=en
\usepackage{setspace}

% Typeset SI units.
% See
%     https://mirror.math.princeton.edu/pub/CTAN/macros/latex/contrib/siunitx/siunitx.pdf
\usepackage{siunitx}
  % See
  %    https://tex.stackexchange.com/questions/611591/bug-in-siunitx-introduced-revealed-by-a-recent-update-micro-produces-unicode-e
  \DeclareSIPrefix{\micro}{\text{\textmu}}{-6}
  % From https://tex.stackexchange.com/questions/513495/problems-with-%C2%B5-displaying-%C5%A3-with-sisetup-option-for-combination-of-siunitx-f
  \DeclareBinaryPrefix\kibi{Ki}{10}
  \DeclareBinaryPrefix\mebi{Mi}{20}
  \DeclareBinaryPrefix\gibi{Gi}{30}
  \DeclareBinaryPrefix\tebi{Ti}{40}
  \DeclareBinaryPrefix\pebi{Pi}{50}
  \DeclareBinaryPrefix\exbi{Ei}{60}
  \DeclareBinaryPrefix\zebi{Zi}{70}
  \DeclareBinaryPrefix\yobi{Yi}{80}
  % See page 61 of
  %     https://mirror.las.iastate.edu/tex-archive/macros/latex/contrib/siunitx/siunitx.pdf
  \DeclareSIUnit \nounit  {\relax}                              % no unit
  \DeclareSIUnit \AA      {\SIUnitSymbolAngstrom}               % \angstrom
  \DeclareSIUnit \arcmin  {\SIUnitSymbolArcminute}              % \arcminute
  \DeclareSIUnit \arcsec  {\SIUnitSymbolArcsecond}              % \arcsecond
  \DeclareSIUnit \au      {au}                                  % \astronomicalunit
  \DeclareSIUnit \b       {b}                                   % \barn
  % Unit macro \bar has been deprecated in release {2021-08-04} {3.0.24}.
  % The BIPM have removed this unit from the SI Brochure.
  % You should define it yourself using \DeclareSIUnit in
  % your source.  The current definition is
  \DeclareSIUnit \bar     {bar}                                 % bar
  \DeclareSIUnit \Bq      {Bq}                                  % \becquerel
  \DeclareSIUnit \B       {B}                                   % \bel
  % Unit macro \bohr has been deprecated in release {2021-08-04} {3.0.24}.
  % The BIPM have removed this unit from the SI Brochure.
  % You should define it yourself using \DeclareSIUnit in
  % your source.  The current definition is
  \DeclareSIUnit\bohr{\text {\ensuremath {a}}_{0}}
  \DeclareSIUnit \bu      {bu}                                  % bushel
  \DeclareSIUnit \cd      {cd}                                  % \candela
  \DeclareSIUnit \C       {C}                                   % \coulomb
  \DeclareSIUnit \Da      {Da}                                  % \dalton
  \DeclareSIUnit \d       {d}                                   % \day
  \DeclareSIUnit \ec      {\text{\ensuremath{e}}}               % \elementarychange
  \DeclareSIUnit \em      {\text{\ensuremath{m_{\textup{e}}}}}  % \electronmasss
  \DeclareSIUnit \ft      {ft}                                  % foot
  \DeclareSIUnit \Gy      {Gy}                                  % \Gy
  \DeclareSIUnit \H       {H}                                   % \henry
  \DeclareSIUnit \ha      {ha}                                  % \hectare
  % Unit macro \hartree has been deprecated in release {2021-08-04} {3.0.24}.
  % The BIPM have removed this unit from the SI Brochure.
  % You should define it yourself using \DeclareSIUnit in
  % your source.  The current definition is
  \DeclareSIUnit \hartree  {\text {\ensuremath {E}}_{\mathrm {h}}}  % \hartree
  \DeclareSIUnit \h       {h}                                   % \hour
  \DeclareSIUnit \hp      {hp}                                  % horsepower
  \DeclareSIUnit \in      {in}                                  % inch
  \DeclareSIUnit \kat     {kat}                                 % \katal
  \DeclareSIUnit \K       {K}                                   % \kelvin
  \DeclareSIUnit \kn      {kn}                                  % \knot
  \DeclareSIUnit \lb      {lb}                                  % pound
  \DeclareSIUnit \lm      {lm}                                  % \lumen
  \DeclareSIUnit \lx      {lx}                                  % \lux
  \DeclareSIUnit \MJ      {MJ}                                  % \mega\joule
  \DeclareSIUnit \mbar    {mbar}                                % \milli\bar
  \DeclareSIUnit \min     {min}                                 % \minute
  % Unit macro \hartree has been deprecated in release {2021-08-04} {3.0.24}.
  % The BIPM have removed this unit from the SI Brochure.
  % You should define it yourself using \DeclareSIUnit in
  % your source.  The current definition is
  \DeclareSIUnit \mmHg    {mmHg}                                % millimeter of mercury
  \DeclareSIUnit \M       {M}                                   % \nauticalmile
  \DeclareSIUnit \Np      {Np}                                  % \neper
  % Unit macro \planckbar has been deprecated in release {2021-08-04} {3.0.24}.
  % The BIPM have removed this unit from the SI Brochure.
  % You should define it yourself using \DeclareSIUnit in
  % your source.  The current definition is
  \DeclareSIUnit \planckbar  {\text {\ensuremath {\hbar}}}      % planckbar
  \DeclareSIUnit \rad     {rad}                                 % \radian
  \DeclareSIUnit \S       {S}                                   % \siemens
  % Unit macro \clight has been deprecated in release {2021-08-04} {3.0.24}.
  % The BIPM have removed this unit from the SI Brochure.
  % You should define it yourself using \DeclareSIUnit in
  % your source.  The current definition is
  \DeclareSIUnit \clight  {\text {\ensuremath {c}}_{0}}         % speed of light
  \DeclareSIUnit \SD      {SD}                                  % standard deviation
  \DeclareSIUnit \Sv      {Sv}                                  % \sievert
  \DeclareSIUnit \sr      {sr}                                  % \steradian
  \DeclareSIUnit \T       {T}                                   % \tesla
  \DeclareSIUnit \t       {t}                                   % \tonne
  \DeclareSIUnit \Wb      {Wb}                                  % \weber
  \DeclareSIUnit \yd      {yd}                                  % yard
  \DeclareSIUnit \y       {y}                                   % year

% Provide an interface to typeset subcaptions.
% See
%     https://ctan.org/pkg/subcaption?lang=en
\usepackage[labelsep=space]{subcaption}
  % \usepackage{url}
  % \usepackage{xkeyval}

% Provide longtable support with possible X column specifier.
% See
%     https://ctan.org/pkg/xltabular?lang=en
\usepackage{xltabular}

% Provide a high-level interface for producing document-level commands.
% See
%     https://ctan.org/pkg/xparse?lang=en
\usepackage{xparse}

% Patch existing commands.
% See
%     https://ctan.org/pkg/xpatch/?lang=en
\usepackage{xpatch}

\ExplSyntaxOn

  % Don't print warning messages about using PDF 1.5, 1.6, and 1.7 files.
  % See
  %   https://tex.stackexchange.com/questions/552143/why-cant-i-set-the-pdf-minor-version-after-loading-beamer-class
  \sys_ensure_backend:  % forces loading of backend, in older systems use \RequirePackage{expl3}
  \pdf_version_gset:n{1.7}

  % Spaces are ignored inside \ExplSyntaxOn.
  % Use ▯ (Unicode U+25AF) instead of \space for a space.
  \catcode`\▯=\active
  \def▯{\space }

  % Define some LaTeX 3 style variants of control sequences..
  \cs_generate_variant:Nn \prop_get:NnNTF {NV}
  \cs_generate_variant:Nn \str_case:nnTF {V}

  % Define \str_case:Vn.
  % The following line doesn't work.
  % \cs_set:Nn {\str_case:Vn} {\str_case:VnTF{#1}{#2}{}{}}
  \def\str_case:Vn#1#2{\str_case:VnTF{#1}{#2}{}{}}

  \newcommand{\ZZAbort}
    {
      \typeout{ABORTING...}
      \typeout{}
      \typeout{}
      \stop
    }

  % ZZ backslash
  % \newcommand{\ZZbs}{\@backslashchar}
  \newcommand{\ZZbs}{\c_backslash_str}
  
  % \zzcheck:nnn <regex> <string to test against regex> <end of error message>
  \cs_set:Npn \zzcheck:nnn #1#2#3
    {
      \regex_match:nnTF
        {#1}
        {#2}
        { \typeout{Match.} }
        {
          \typeout{No▯match.}
          % \remember contains the '~'
          \typeout{\ZZbs\remember is▯'#2'.}
          \typeout{It▯should▯be▯'#3'.}
          \stop
        }
    }

  \cs_generate_variant:Nn \zzcheck:nnn {nv}

  \cs_set:Npn \ZZCheck:nnn #1#2#3
    {
      \def\remember{#2~}
      \zzcheck:nvn {#1} {#2} {#3}
    }


  \newcommand{\ZZDebuggingMessage}{Debugging...}
  \newcounter{ZZcountersavedpagenumber}
  \newcounter{ZZprevindex}
  \newcommand{\ZZcommittee}{\relax}
  \newcommand{\ZZkeywords}{\relax}
  \newcommand{\ZZprogramhead}{\relax}
  \newcommand{\ZZabbreviationsname}{ABBREVIATIONS}
  \newcommand{\ZZabstractname}{ABSTRACT}
  \newcommand{\ZZacknowledgmentsname}{ACKNOWLEDGMENTS}
  \renewcommand{\baselinestretch}{1.5}
  \newcommand{\ZZchapterblankpages}{false}
  \newcommand{\ZZcoversheets}{false}
  \newcommand{\ZZendnotes}{false}
  \newcommand{\ZZfrenchspacing}{false}
  \newcommand{\ZZglossaryname}{GLOSSARY}
  \newcommand{\ZZiso}{false}
  \newcommand{\ZZlistoffiguresname}{LIST~OF~FIGURES}
  \newcommand{\ZZlistoflistingsname}{LIST~OF~LISTINGS}
  \newcommand{\ZZlistofprotocolsname}{LIST~OF~PROTOCOLS}
  \newcommand{\ZZlistofschemesname}{LIST~OF~SCHEMES}
  \newcommand{\ZZlistoftablesname}{LIST~OF~TABLES}
  \newcommand{\ZZnomenclaturename}{NOMENCLATURE}
  \newcommand{\ZZoutput}{pdf}
  \newcommand{\ZZprefacename}{PREFACE}
  \newcommand{\ZZsymbolsname}{LIST~OF~SYMBOLS}
  \newcommand{\ZZtableofcontentsname}{TABLE~OF~CONTENTS}
  \newcommand{\ZZvitaname}{VITA}

  %  CONFIGURE THE BIBLIOGRAPHY
  %
  %  Search for "REQUIRED PARAMETERS" above.  Mentally compute the
  %  string consisting of your INSTITUTION CODE, followed by a "-",
  %  followed by your CAMPUS CODE, followed by a "-", followed by your
  %  PROGRAM CODE (e.g., "pu-wl-aae").
  %
  %  Define \ConfigureBibliography.
  %
  % BIBLATEX OPTION                   DESCRIPTION
  % backend         = biber           Use biber to process the bibliography.
  % bibstyle        = ...             Use the ... bibliography style.
  % citestyle       = numeric-comp    Change [1],[3] to [1,3] and [1]--[3] to [1--3].
  % dashed          = false           Print repeated names---don't use a dash for repeated names.
  % locallabelwidth = true            Use locally calculated (for this bibliography)
  %                                   or global value calculated for all bibliographies.
  % sortcites       = true            Sort multiple citations.
  %

  % \ZZoverridebibstyle AND \ZZoverridecitestyle DON'T WORK YET.
  % DO NOT TRY TO USE THEM.
  % Define \ZZbibstyleoverride_str and \ZZcitestyleoverride_str.
  % Those can be overridden using
  %     \ZZoverridebibstyle
  % and
  %     \ZZoverridecitestyle
  % in the thesis.tex file.
  \str_new:N \ZZbibstyleoverride_str
  \str_new:N \ZZcitestyleoverride_str

  % Based on page 99 of
  %   https://ctan.math.illinois.edu/macros/latex/contrib/biblatex/doc/biblatex.pdf
  % See page 113 of LaTeX User's Guide and Reference Manual, second edition,
  % for list dimensions diagram.
  \newcommand{\HangIndent}[1]
  {
    \setlength{\itemindent}{-#1}
    \setlength{\leftmargin}{#1}
  }
  
  % \ZZoverridebibstyle overrides the bibstyle and citestyle.
  % \ZZoverridecitestyle overrides the citestyle.
  \def\ZZoverridebibstyle#1{\str_set:Nn  \ZZbibstyleoverride_str  {#1}}
  \def\ZZoverridecitestyle#1{\str_set:Nn \ZZcitestyleoverride_str {#1}}

  \newcommand\ConfigureBibliography
    {

      \def\ZZsetbibandcitestyle##1
        {
          \def\ZZibibstyle{##1}   \def\ZZbibstyle{##1}
          \def\ZZicitestyle{##1}  \def\ZZcitestyle{##1}
          % Process \ZZbibstyleoverride_str.
          \str_if_empty:NTF
            \ZZbibstyleoverride_str
            % TRUE CODE
            {
            }
            % FALSE CODE
            {
              \def\ZZbibstyle{\str_use:N \ZZbibstyleoverride_str}
              \def\ZZcitestyle{\str_use:N \ZZbibstyleoverride_str}
            }
          % Process \ZZcitestyleoverride_str.
          \str_if_empty:NTF
            \ZZcitestyleoverride_str
            % TRUE CODE
            {
            }
            % FALSE CODE
            {
              \def\ZZcitestyle{str_use:N \ZZcitestyleoverride_str}
            }
        }

      \newcommand{\ZZamabib}
        {
          \ZZsetbibandcitestyle{ama}
          \usepackage[
            backend         = biber,
            citestyle       = \ZZcitestyle,
            locallabelwidth = true,
            sortcites       = true,
            style           = \ZZbibstyle,
          ]{biblatex}
          \addbibresource{all-biblatex.bib}
          % Allow bibliography line break in URLs and DOIs after all lowercase and uppercase letters.
          % See
          %     https://tex.stackexchange.com/questions/134191/line-breaks-of-long-urls-in-biblatex-bibliography
          \setcounter{biburllcpenalty}{7000}
          \setcounter{biburlucpenalty}{8000}
        }

      \newcommand{\ZZapabib}
        {
          \ZZsetbibandcitestyle{apa}
          \usepackage[
            backend         = biber,
            citestyle       = \ZZcitestyle,
            dashed          = false,
            locallabelwidth = true,
            sortcites       = true,
            style           = \ZZbibstyle,
          ]{biblatex}
          \addbibresource{all-biblatex.bib}
          % Allow bibliography line break in URLs and DOIs after all lowercase and uppercase letters.
          % See
          %     https://tex.stackexchange.com/questions/134191/line-breaks-of-long-urls-in-biblatex-bibliography
          \setcounter{biburllcpenalty}{7000}
          \setcounter{biburlucpenalty}{8000}
        }

      \newcommand{\ZZauthoryearbib}
        {
          \ZZsetbibandcitestyle{authoryear}
          \usepackage[
            backend         = biber,
            bibstyle        = \ZZbibstyle,
            citestyle       = \ZZcitestyle,
            dashed          = false,
            locallabelwidth = false,
          ]{biblatex}
          \addbibresource{all-biblatex.bib}
          % Allow bibliography line break in URLs and DOIs after all lowercase and uppercase letters.
          % See
          %     https://tex.stackexchange.com/questions/134191/line-breaks-of-long-urls-in-biblatex-bibliography
          \setcounter{biburllcpenalty}{7000}
          \setcounter{biburlucpenalty}{8000}
        }

      \newcommand{\ZZchicagobib}
        {
          \ZZsetbibandcitestyle{chicago}
          \usepackage[
            backend = biber,
            notes,
          ]{biblatex-chicago}
          \addbibresource{all-biblatex.bib}
          % 
          \AtEveryBibitem{% Clean up the bibtex rather than editing it
            \ifentrytype{software}
              {}
              {\clearfield{urlyear}\clearfield{urlmonth}\clearfield{urlday}}
          }
          % Allow bibliography line break in URLs and DOIs after all lowercase and uppercase letters.
          % See
          %     https://tex.stackexchange.com/questions/134191/line-breaks-of-long-urls-in-biblatex-bibliography
          \setcounter{biburllcpenalty}{7000}
          \setcounter{biburlucpenalty}{8000}
        }

      \newcommand{\ZZieeebib}
        {
          \ZZsetbibandcitestyle{ieee}
          \usepackage[
            backend         = biber,
            bibstyle        = \ZZbibstyle,
            citestyle       = numeric-comp,
            dashed          = false,
            locallabelwidth = true,
            sortcites       = true,
          ]{biblatex}
          \addbibresource{all-biblatex.bib}
          % Allow bibliography line break in URLs and DOIs after all lowercase and uppercase letters.
          % See
          %     https://tex.stackexchange.com/questions/134191/line-breaks-of-long-urls-in-biblatex-bibliography
          \setcounter{biburllcpenalty}{7000}
          \setcounter{biburlucpenalty}{8000}
        }

      \newcommand{\ZZnumericbib}
        {
          \ZZsetbibandcitestyle{numeric}
          \usepackage[
            backend         = biber,
            bibstyle        = \ZZbibstyle,
            citestyle       = \ZZcitestyle,
            locallabelwidth = true,
            sortcites       = true,
          ]{biblatex}
          \addbibresource{all-biblatex.bib}
          % Allow bibliography line break in URLs and DOIs after all lowercase and uppercase letters.
          % See
          %     https://tex.stackexchange.com/questions/134191/line-breaks-of-long-urls-in-biblatex-bibliography
          \setcounter{biburllcpenalty}{7000}
          \setcounter{biburlucpenalty}{8000}
        }

      \str_case:VnTF
        {\ZZinscampro}
        {
          {pu-fw-ap}      {\ZZieeebib}
          {pu-fw-bio}     {\ZZieeebib}
          {pu-fw-cme?}    {\ZZieeebib}
          {pu-fw-comm}    {\ZZieeebib}
          {pu-fw-cs}      {\ZZieeebib}
          {pu-fw-ece}     {\ZZieeebib}
          {pu-fw-engl}    {\ZZapabib}
          {pu-fw-phys}    {\ZZieeebib}
          {pu-fw-sped}    {\ZZieeebib}
          {pu-ha-bio}     {\ZZieeebib}
          {pu-ha-cit}     {\ZZieeebib}
          {pu-ha-ece}     {\ZZieeebib}
          {pu-ha-hdfs}    {\ZZieeebib}
          {pu-ha-hist}    {\ZZapabib}
          {pu-ha-mce}     {\ZZieeebib}
          {pu-in-bio}     {\ZZieeebib}
          {pu-in-bme}     {\ZZieeebib}
          {pu-in-ccb?}    {\ZZieeebib}
          {pu-in-cis?}    {\ZZieeebib}
          {pu-in-ece}     {\ZZieeebib}
          {pu-in-fis?}    {\ZZieeebib}
          {pu-in-math}    {\ZZieeebib}
          {pu-in-me}      {\ZZieeebib}
          {pu-in-mste}    {\ZZieeebib}
          {pu-in-neu?}    {\ZZieeebib}
          {pu-in-phys}    {\ZZieeebib}
          {pu-we-bio}     {\ZZieeebib}
          {pu-we-ece}     {\ZZieeebib}
          {pu-we-hdfs}    {\ZZieeebib}
          {pu-we-hist}    {\ZZapabib}
          {pu-we-mce}     {\ZZieeebib}
          % Jon Mrozinski, Graduate Program Administrator, School of Aeronautics and Astronautics wrote on 2024-04-01
          %   After conferring in our department and School, we
          %   determined that AAE does not have a preferred
          %   Reference/Bibliography/Citation style that is applied to
          %   all AAE theses. Different styles have been seen used on
          %   students' theses, as they usually follow what is used
          %   for the journals where they publish their work. With six
          %   different major disciplines, the journals they favor can
          %   vary, and the different journals have different style
          %   requirements.
          % Mark Senn <mark@purdue.edu> left IEEE as their default on 2024-03-30.
          {pu-wl-aae}     {\ZZieeebib}
          % Nikki Zimmerman <nzimmerm@purdue.edu>, Graduate Program Administrator, Agricultural and Biological Engineering, wrote on 2024-04-11:
          %   We are good with APA 7. Confirmed with our head.
          {pu-wl-abe}     {\ZZapabib}
          % Minde Freischlag, Graduate Program Assistant for Art & Design, Rueff School of Design, Art & Performance wrote on 2024-05-07:
          %   Please change the writing [sic -mark] style for Art & Design to Chicago.
          {pu-wl-ad}      {\ZZchicagobib}
          % Nicole Widmar wrote on 2023-05-13:
          %   [Authoryear is fine -mark] for the default, as
          %   long as we are clear that while it might be
          %   the default - our Department does not have a
          %   required format.
          {pu-wl-agecon}  {\ZZauthoryearbib}
          {pu-wl-agry}    {\ZZieeebib}
          {pu-wl-amst}    {\ZZieeebib}
          {pu-wl-ansc}    {\ZZieeebib}
          {pu-wl-anth}    {\ZZieeebib}
          % Diane Greenberg <dgreenb@purdue.edu>, Graduate Program Coordinator, Agricultural Sciences Education and Communication wrote on 2024-03-28
          %   In the Agricultural Sciences Education and Communication department we use
          %   the APA - 7th Edition when it comes to references /bibliography/citations.
          {pu-wl-asec}    {\ZZapabib}
          {pu-wl-bchm}    {\ZZieeebib}
          {pu-wl-bcm}     {\ZZieeebib}
          {pu-wl-biol}    {\ZZieeebib}
          % Tammy Siemers <tsiemers@purdue.edu>, wrote on 2024-03-29
          %   In Biomedical Engineering, we prefer either IEEE or APA.
          {pu-wl-bme}     {\ZZieeebib}
          % Susan M. Mendrysa <mendrysa@purdue.edu>, PhD, Professor & Interim Dept Head of Basic Medical Sciences wrote on 2024-04-03:
          %   Please indicate Chicago style as preferred for
          %   BMS. However, BMS does not require a specific reference
          %   style as the standard varies by sub-discipline within
          %   the dept.  We only require the use of a consistent style
          %   throughout the thesis bibliography.
          {pu-wl-bms}     {\ZZchicagobib}
          {pu-wl-btny}    {\ZZieeebib}
          {pu-wl-cce}     {\ZZieeebib}
          {pu-wl-cgt}     {\ZZieeebib}
          % Bev Johnson <bevjohnson@purdue.edu>, Graduate Program Administrator, Chemical Engineering wrote on 2024-03-28
          %   Do we have to choose a preference?  I don't think that our faculty are particular.
          % Mark Senn <mark@purdue.edu> left IEEE as their default on 2024-03-30.
          {pu-wl-che}     {\ZZieeebib}
          % Jordan Taylor Harris <harri698@purdue.edu>, Main Office Assistant & Formatting Advisor, Department of Chemistry wrote on 2024-03-29
          %   The preferred citation style for Chemistry is ACS style.
          {pu-wl-chem}    {\ZZchemacsbib}
          % Joy Kane, Graduate Programs Assistant, Computer and Information Technology wrote on 2024-03-28
          %   Computer and Information Technology Department prefers IEEE as first choice and APA as second choice.
          {pu-wl-cit}     {\ZZieeebib}
          {pu-wl-cmpl}    {\ZZieeebib}
          {pu-wl-com}     {\ZZieeebib}
          {pu-wl-cpb}     {\ZZieeebib}
          {pu-wl-cs}      {\ZZieeebib}
          {pu-wl-csr}     {\ZZieeebib}
          % Emjai Gregory <egregor@purdue.edu>, EAPS Graduate Coordinator, wrote on 2024-03-28
          %   The APA will be fine.
          {pu-wl-eaps}    {\ZZapabib}
          {pu-wl-ece}     {\ZZieeebib}
          {pu-wl-econ}    {\ZZieeebib}
          {pu-wl-edci}    {\ZZieeebib}
          {pu-wl-edst}    {\ZZieeebib}
          % Michelle Indarjit <mindarji@purdue.edu>, Graduate Administrative Assistant, Environmental and Ecological Engineering, wrote on 2024-03-29
          %   EEE does not have a preferred style preference in reference to: references/bibliography/citations.
          %   Students and advisors will decide what they would prefer.
          {pu-wl-eee}     {\ZZieeebib}
          % Loretta McKinniss <lmckinni@purdue.edu>, Graduate Program Administrative Assistant, School of Engineering Education wrote on 2024-03-28
          %   The School of Engineering Education follows APA.
          {pu-wl-ene}     {\ZZapabib}
          {pu-wl-engl}    {\ZZapabib}
          % Molly Weber <weber77@purdue.edu>, Administrative Assistant,
          % Department of Entomology wrote on 2024-04-15:
          %   The chair said "As far as I'm aware, we've never had a set
          %   department style for thesis submission." Can our dept thesis
          %   template state an [any -mark] style is acceptable?
          {pu-wl-entm}    {\ZZapabib}
          % Duane D. Dunlap <ddunlap@purdue.edu>, Prof. Dr. and Chair of Graduate Programs, School of Engineering Technology wrote on 2024-03-28
          %   The School of Engineering Technology uses the attached document SoET Thesis/Directed Project Checklist for format.
          % From
          % School of Engineering Technology Thesis and Directed Project Checklist (approved 10/16/15)
          % https://polytechnic.purdue.edu/sites/default/files/files/SoET%20Thesis%20and%20Directed%20Project%20Format%20Checklist%20-%2010-21-15.pdf
          %   Full citations in must be in either APA (preferred) or IEEE format.
          {pu-wl-et}      {\ZZapabib}
          {pu-wl-fnr}     {\ZZieeebib}
          % Betty Lewis wrote on 2024-03-28:
          %    Food Science’s preferred style is APA.
          {pu-wl-fs}      {\ZZapabib}
          % Elizabeth Atha, Graduate Program Coordinator, Department of 
          % Human Development and Family Science wrote on 2024-04-22:
          %   The Department of HDFS prefers APA.
          {pu-wl-hdfs}    {\ZZapabib}
          % O.T. Ford <oford@purdue.edu>, Academic Program Manager,
          % Department of History, wrote on 2024-04-01:
          %   The Department of History has no preference for the
          %   style of references used in theses and dissertations.
          %   Students can determine that with their committees.
          %   Is that an option?
          {pu-wl-hist}    {\ZZapabib}
          {pu-wl-hk}      {\ZZieeebib}
          {pu-wl-hort}    {\ZZapabib}
          {pu-wl-hsci}    {\ZZieeebib}
          {pu-wl-htm}     {\ZZieeebib}
          {pu-wl-ibsc}    {\ZZieeebib}
          {pu-wl-ie}      {\ZZieeebib}
          {pu-wl-ipph}    {\ZZieeebib}
          {pu-wl-lc}      {\ZZieeebib}
          {pu-wl-ling}    {\ZZieeebib}
          % Shannon Cassady, Graduate Coordinator, Math Department,
          % wrote on 2024-04-01:
          %   MLA, Chicago, IEEE are all fine for the math department.
          {pu-wl-math}    {\ZZieeebib}
          {pu-wl-mcmp}    {\ZZieeebib}
          % Sheri Tague <sltague@purdue.edu>, Sr. Graduate Program Manager,
          % Mechanical Engineering wrote on 2024-03-28:
          %   Mechanical Engineering doesn't have a preference.
          {pu-wl-me}      {\ZZieeebib}
          {pu-wl-mgmt}    {\ZZieeebib}
          % Kevin Trumble <trumble@purdue.edu>, Graduate Study chair wrote ~2024-03-29
          %   The student figures out what style is appropriate for the discipline or
          %   sub-discipline of their work.  We have some who use ACS, others IEEE,
          %   others other styles, we don't have one.  The current template has been
          %   working well for our students.
          {pu-wl-mse}     {\ZZieeebib}
          % Brad McDaniel <bmmcdani@purdue.edu>, Academic Programs
          % Administrator, School of Nuclear Engineering wrote on 2024-03-28:
          %   The School of Nuclear Engineering's preferred style is APA.
          {pu-wl-ne}      {\ZZapabib}
          {pu-wl-nur}     {\ZZieeebib}
          {pu-wl-nutr}    {\ZZieeebib}
          {pu-wl-obhr}    {\ZZieeebib}
          {pu-wl-owl}     {\ZZieeebib}
          {pu-wl-phillit} {\ZZieeebib}
          % Vickie A. Sanders <sanders@purdue.edu> wrote on 2024-03-28:
          %   For Philosophy Department either APA or Chicago.
          %   As long as those are an option, that is fine.
          {pu-wl-phil}    {\ZZapabib}
          {pu-wl-phpr}    {\ZZieeebib}
          {pu-wl-phys}    {\ZZieeebib}
          % POL Grad Coordinator <polgradcoordinator@purdue.edu>,
          % Department of Political Science wrote on 2024-03-28:
          %   ... Political Science chooses the Chicago style.
%         {pu-wl-pol}     {\ZZapabib}
          {pu-wl-psd}     {\ZZieeebib}
          % Nancy O'Brien <nobrien@purdue.edu>, Program Administration Specialist and Graduate Coordinator, Department of Psychological Sciences, wrote on 2024-03-28
          %   ...the Department of Psychological Sciences prefers APA style for references, bibliography, and citations.
          {pu-wl-psy}     {\ZZapabib}
          % Shauna Stapleton <sstaplet@purdue.edu>, Academic Programs Manager,
          % Department of Public Health, wrote on 2024-04-01:
          %   The department of public health would like graduate
          %   students to have the option to choose between APA and
          %   AMA for their submissions. Is this possible?
          {pu-wl-pubh}    {\ZZapabib}
          % John H. Mott <jhmott@purdue.edu> wrote on 2024-05-20:
          %   APA 7th.
          {pu-wl-satt}    {\ZZapabib}
          {pu-wl-slhs}    {\ZZieeebib}
          {pu-wl-soc}     {\ZZieeebib}
          % Patti Foster, Graduate Coordinator, Department of Statistics,
          % wrote on 2024-03-28:
          %   ...use bibliography style of Chicago...
          {pu-wl-stat}    {\ZZchicagobib}
          {pu-wl-tdo}     {\ZZieeebib}
          {pu-wl-tech}    {\ZZieeebib}
          {pu-wl-thtr}    {\ZZieeebib}
          {pu-wl-tli}     {\ZZapabib}
          % Jesse Mabbitt, Graduate and International Programs
          % Coordinator, Veterinary Clinical Sciences, College of
          % Veterinary Medicine wrote on 2024-04-12:
          %   The Veterinary Clinical Sciences Department does not
          %   have a style preference when it comes to
          %   reference/bibliography/citations.  If the Purdue
          %   LaTeX/Overleaf thesis template has a specific style
          %   listed that can remain the same.
          {pu-wl-vcs}     {\ZZamabib}
          {pu-wl-ydae}    {\ZZieeebib}
          {uh-manoa-econ} {\ZZieeebib}
        }
        % TRUE CODE
        { }
        % FALSE CODE
        { }
    }

  %
  %  Define \PrintBibliography.
  %

  % See
  %     https://tex.stackexchange.com/questions/198551/biblatex-using-shorthands-and-numeric-style-label-sizes#:~:text=As%20mentioned%20in%20Biblatex%2C%20Use%20of%20prefixnumber%20changes,makes%20the%20labelwidth%20calculation%20local%20for%20each%20printbibliography.
  % for how to control labelnumberwidth on a per bibliography basis.
  %

  \newcommand\PrintBibliography
    {
      \makeatletter
        \setlength{\leftmargini}{0pt}
        \defbibenvironment{bibliography}
          {
            \list
            {
              \printtext
                [labelnumberwidth]
                {
                  \printfield{prefixnumber}
                  \printfield{labelnumber}
                }
            }
            {
              \str_case:NnTF
                {\ZZbibstyle}
                {
                  {ama}
                  {
                    % See
                    % https://owl.purdue.edu/owl/research_and_citation/ama_style/index.html
                    % and the 'FOR IEEE style bibliography.' section below.
                    \settowidth{\labelwidth}{12.}
                    \setlength{\labelsep}{1em}
                    \setlength{\itemsep}{0pt}
                    \setlength{\parsep}{12pt}
%%%                 \renewcommand*{\makelabel}[1]{##1\hfil}
                  }
                  {apa}
                  {
                    \HangIndent{0.5truein}
                    % Lisa Williams <will2922@purdue.edu> of the Graduate School
                    % wrote on 2024-03-05 at 11:24:
                    %     ...  Spacing between references and publications for all departments
                    %     and programs should be no less than 12-point.  Thank you!  ...
                    \setlength{\itemsep}{0pt}
                    \setlength{\parsep}{12pt}
                  }
                  {authoryear}
                  {
%                    % See page 113 of LaTeX User's Guide and Reference Manual,
%                    % Leslie Lamport, second edition, November, 1994.
%                    \setlength{\itemindent}{0pt}
%                    \setlength{\labelsep}{1em}
%                    \settowidth{\labelwidth}{[12]}
%                    \setlength{\listparindent}{0pt}
%                    \setlength{\rightmargin}{0pt}
%                    \setlength{\leftmargin}{\labelwidth}
%                      \addtolength{\leftmargin}{\labelsep}
%                    % From page 112 and 114 of LaTeX User's Guide and Reference Manual,
%                    % Leslie Lamport, second edition, November, 1994.
%                    \setlength{\itemsep}{0pt}
%                    \setlength{\parsep}{12pt}
%                    % Outer (level 0) \def uses one #, level 1 uses two #,
%                    % level 2 uses four #---each new level uses twice the
%                    % number of # as previous level.
%                    \renewcommand*{\makelabel}[1]{####1\hfil}
                    \HangIndent{0.5truein}
                    % Lisa Williams <will2922@purdue.edu> of the Graduate School
                    % wrote on 2024-03-05 at 11:24:
                    %     ...  Spacing between references and publications for all departments
                    %     and programs should be no less than 12-point.  Thank you!  ...
                    \setlength{\itemsep}{0pt}
                    \setlength{\parsep}{12pt}
                  }
                  {chicago}
                  {
                    \HangIndent{0.5truein}
                    % Lisa Williams <will2922@purdue.edu> of the Graduate School
                    % wrote on 2024-03-05 at 11:24:
                    %     ...  Spacing between references and publications for all departments
                    %     and programs should be no less than 12-point.  Thank you!  ...
                    \setlength{\itemsep}{0pt}
                    \setlength{\parsep}{12pt}
                  }
                  {ieee}
                  {
                    % See page 113 of LaTeX User's Guide and Reference Manual,
                    % Leslie Lamport, second edition, November, 1994 for list
                    % format parameters.  See page 114, the paragraph starting
                    % with 'The label is typeset in LR mode.' for what happens
                    % when the label is wider than \labelwidth.
                    \settowidth{\labelwidth}{[12]}
                    \setlength{\labelsep}{1em}
                    \setlength{\leftmargin}{\labelwidth}
                      \addtolength{\leftmargin}{\labelsep}
                    \setlength{\parsep}{6pt plus 1pt minus 1pt}
                    \setlength{\itemsep}{0pt}
                    \setlength{\parsep}{12pt}
                    % Outer (level 0) \def uses one #, level 1 uses two #,
                    % level 2 uses four #---each new level uses twice the
                    % number of # as previous level.
%%%                    \renewcommand*{\makelabel}[1]{##1\hfil}
                  }
                  {numeric}
                  {
                    % See page 113 of LaTeX User's Guide and Reference Manual,
                    % Leslie Lamport, second edition, November, 1994 for list
                    % format parameters.  See page 114, the paragraph starting
                    % with 'The label is typeset in LR mode.' for what happens
                    % when the label is wider than \labelwidth.
                    \settowidth{\labelwidth}{[12]}
                    \setlength{\labelsep}{1em}
                    \setlength{\leftmargin}{\labelwidth}
                      \addtolength{\leftmargin}{\labelsep}
                    \setlength{\parsep}{6pt plus 1pt minus 1pt}
                    \setlength{\itemsep}{0pt}
                    \setlength{\parsep}{12pt}
                    % Outer (level 0) \def uses one #, level 1 uses two #,
                    % level 2 uses four #---each new level uses twice the
                    % number of # as previous level.
%%%                    \renewcommand*{\makelabel}[1]{##1\hfil}
                  }
                }
                % TRUE CODE
                {
                  \typeout{Code▯for▯\ZZbs ZZbibstyle▯'\ZZbibstyle'▯found.}
                }
                % FALSE CODE
                {
                  \typeout{No▯code▯for▯\ZZbs ZZbibstyle▯'\ZZbibstyle'▯found.}
                  \ZZAbort
                }
            }
          }
          {\endlist}
          {\item}
      \makeatother
      \begin{singlespace}
        % From
        % https://tex.stackexchange.com/questions/17360/reduce-spacing-in-bibliography-using-biblatex
        \bibitemsep = 0pt  % length between two different entries
        \bibnamesep = 0pt  % length between two entries of different authors
        \bibinitsep = 0pt  % length between two entries of authors with different first letter
        % Do not split a bibliography entry between pages.
        % See
        %     https://tex.stackexchange.com/questions/43260/biblatex-no-pagebreak-in-bibliography-entry
        % From
        % https://tex.stackexchange.com/questions/152773/please-tutor-the-usage-of-patchcmd-and-xpatch
        %     I would consider patchcmd and \patchcommand as obsolete: with
        %     etoolbox the complete functionality of \patchcommand can be
        %     obtained with \pretocmd and \apptocmd and much more is possible
        %     with \patchcmd.
        % From
        % "The xpatch package Extending etoolbox patching commands"
        % http://ctan.math.washington.edu/tex-archive/macros/latex/contrib/xpatch/xpatch.pdf
        %     The well known etoolbox package provides a bunch of functions
        %     for patching existing commands; in particular \patchcmd,
        %     \pretocmd and \apptocmd that do a wonderful job, but suffer
        %     from a limitation:
        % This uses the \xpatchcmd command in the xpatch package.
        % Change \interlinepenalty=5000 to \interlinepenalty=10000 in \bibsetup.
        \xpatchcmd
          \bibsetup
          {\interlinepenalty=5000}
          {\interlinepenalty=10000}
          {\typeout{changed▯\interlinepenalty▯to▯10000}}
          {\typeout{can't▯change▯\interlinepenalty▯to▯10000}\stop}
        \printbibliography
          [
            heading         = {bibintoc},    % Put an entry in the table of contents.
            locallabelwidth = {true},
            title           = {REFERENCES},  % The title of the bibliography should be "REFERENCES".
            % block         = {nbpar},
          ]
        \endgraf
      \end{singlespace}
    }


  %
  %  Define \PrintChapterBibliography.
  %

  \def\PrintChapterBibliography
    {
      \defbibenvironment{bibliography}
        {
          \list
            {
              \printtext[labelnumberwidth]
              {
                \printfield{prefixnumber}
                \printfield{labelnumber}
              }
            }%
            {
              % Bibliography entries must not be split between pages.
              \interlinepenalty=10000
              \setlength{\bibhang}{0pt}
              % In a reference that begins with "[10] L. E. van Dijk"
              % it is 0.33in from the left edge of the paper to "L".
              \setlength{\itemindent}{0.33in}
              \setlength{\itemsep}{\bibitemsep}
              \setlength{\leftmargin}{0pt}
              \setlength{\parsep}{\bibparsep}
              \setlength{\rightmargin}{0pt}
            }
        }
        {\endlist}
        {\item}
      \printbibliography[heading=subbibintoc]
    }

  \ZZCheck:nnn {\A.+\Z} {ZZauthor} {one or more characters}


  % \ZZmake defines \ZZadd and \ZZcheck.
  %
  % For examplee, doing \ZZmake{ins}{institution}
  %     o   defines \l_ins_prop and \l_ins_seq
  %     o   creates \ZZadd to add to \l_ins_prop and \l_ins_seq
  %     o   creates \ZZcheck to check if the \ZZinstitution is legal
  \def\ZZmake#1#2
    {
      \prop_new:c {l_#1_prop}
      \seq_new:c  {l_#1_seq}
      \def\ZZadd##1##2
        {
          \prop_put:cnn     {l_#1_prop} {##1} {##2}
          \seq_put_right:cn {l_#1_seq}  {##1}
        }
      \def\ZZcheck
        {
          \xdef\temp{\csname ZZ#2\endcsname}
          \prop_get:cVNTF
            {l_#1_prop}
            \temp  % don't know how to use, for example, "{ZZ#2}" here offhand
            \l_tmpb_tl
            % TRUE CODE
            {
              \typeout{Setting▯\ZZbs ZZ#1▯to▯"\l_tmpb_tl".}
              \tl_new:c     {ZZ#1}
              \tl_set_eq:cN {ZZ#1} \l_tmpb_tl
            }
            % FALSE CODE
            {
              \typeout{Bad▯"\ZZbs ZZ#2"▯option.}
              \typeout{"\csname ZZ#2\endcsname"▯is▯not▯on▯this▯list:}
              \seq_map_inline:cn {l_#1_seq} {\typeout{▯▯▯▯####1}}
              \ZZAbort
            }
        }
    }

  \ZZmake {cam} {campus}
  \ZZadd  {Bloomington}    {bl}
  \ZZadd  {Fort~Wayne}     {fw}
  \ZZadd  {Hammond}        {ha}
  \ZZadd  {Indianapolis}   {in}
  \ZZadd  {Manoa}          {ma}
  \ZZadd  {Reno}           {re}
  \ZZadd  {Westville}      {we}
  \ZZadd  {West~Lafayette} {wl}
  \ZZcheck

  \ZZmake {deg} {degree}
  \ZZadd {Doctor~of~Audiology}                                          {aud}
  \ZZadd {Doctor~of~Nursing~Practice}                                   {dnp}
  \ZZadd {Doctor~of~Philosophy}                                         {phd}
  \ZZadd {Doctor~of~Technology}                                         {dtech}
  \ZZadd {Master~of~Arts}                                               {ma}
  \ZZadd {Master~of~Fine~Arts}                                          {mfa}
  \ZZadd {Master~of~Library~Science}                                    {mls}
  \ZZadd {Master~of~Science}                                            {ms}
  \ZZadd {Master~of~Science~in~Aeronautics~and~Astronautics}            {msaa}
  \ZZadd {Master~of~Science~in~Agriculture}                             {msa}
  \ZZadd {Master~of~Science~in~Agricultural~and~Biological~Engineering} {msabe}
  \ZZadd {Master~of~Science~in~Aviation~and~Aerospace~Management}       {msaam}
  \ZZadd {Master~of~Science~in~Biomedical~Engineering}                  {msbme}
  \ZZadd {Master~of~Science~in~Building~Construction~Management}        {msbcm}
  \ZZadd {Master~of~Science~in~Chemical~Engineering}                    {msche}
  \ZZadd {Master~of~Science~in~Civil~Engineering}                       {msce}
  \ZZadd {Master~of~Science~in~Education}                               {msed}
  \ZZadd {Master~of~Science~in~Electrical~and~Computer~Engineering}     {msece}
  \ZZadd {Master~of~Science~in~Engineering}                             {mse}
  \ZZadd {Master~of~Science~in~Engineering~Technology}                  {mset}
  \ZZadd {Master~of~Science~in~Human~Resources~Management}              {mshrm}
  \ZZadd {Master~of~Science~in~Industrial~Engineering}                  {msie}
  \ZZadd {Master~of~Science~in~Industrial~Technology}                   {msit}
  \ZZadd {Master~of~Science~in~Materials~Engineering}                   {msmse}
  \ZZadd {Master~of~Science~in~Mechanical~Engineering}                  {msme}
  \ZZadd {Master~of~Science~in~Nuclear~Engineering}                     {msne}
  \ZZadd {Ph.D.~in~Technology}                                          {phdtech}
  \ZZcheck

  \ZZCheck:nnn {\A.+\Z} {ZZcommittee} {one or more characters}

  \ZZmake {doc} {document}
  \ZZadd {A~Directed~Research~Project} {adrp}
  \ZZadd {A~Dissertation}              {dissertation}
  \ZZadd {A~Master's~Bypass~Report}    {ambr}
  \ZZadd {A~Preliminary~Report}        {prelim}
  \ZZadd {A~Thesis}                    {thesis}
  \ZZcheck

  \ZZCheck:nnn {\A(August|May|December)\ 20[2-9][0-9]\Z} {ZZgraduation} {should~be~'August'~or~'May'~or~'December' followed by 2020 through 2099}

  \ZZmake {ins} {institution}
  \ZZadd {Indiana~University}   {iu}
  \ZZadd {Purdue~University}    {pu}
  \ZZadd {University~of~Hawaii} {uh}
  \ZZcheck

  \ZZCheck:nnn {\A.+\Z} {ZZkeywords} {one or more characters}

  \ZZmake {pro} {program}
  \ZZadd {Aeronautics~and~Astronautics}                          {aae}
  \ZZadd {Agricultural~Economics}                                {agecon}
  \ZZadd {Agricultural~and~Biological~Engineering}               {abe}
  \ZZadd {Agricultural~Sciences~Education~and~Communication}     {asec}
  \ZZadd {Agronomy}                                              {agry}
  \ZZadd {American~Studies}                                      {amst}
  \ZZadd {Animal~Sciences}                                       {ansc}
  \ZZadd {Anthropology}                                          {anth}
  \ZZadd {Applied~Physics}                                       {ap}
  \ZZadd {Art~and~Design}                                        {ad}
  \ZZadd {Aviation~and~Transportation~Technology}                {satt}
  \ZZadd {Basic~Medical~Sciences}                                {bms}
  \ZZadd {Biochemistry}                                          {bchm}
  \ZZadd {Biological~Sciences}                                   {biol}
  \ZZadd {Biology}                                               {bio}
  \ZZadd {Biomedical~Engineering}                                {bme}
  \ZZadd {Botany~and~Plant~Pathology}                            {btny}
  \ZZadd {Chemical~Engineering}                                  {che}
  \ZZadd {Chemistry}                                             {chem}
  \ZZadd {Chemistry~and~Chemical~Biology}                        {ccb?}
  \ZZadd {Civil~and~Mechanical~Engineering}                      {cme?}
  \ZZadd {Civil~and~Construction~Engineering}                    {cce}
  \ZZadd {Communication}                                         {comm}
  \ZZadd {Communications}                                        {com}
  \ZZadd {Comparitive~Pathobiology}                              {cpb}
  \ZZadd {Computer~and~Information~Science}                      {cis?}
  \ZZadd {Computer~and~Information~Technology}                   {cit}
  \ZZadd {Computer~Graphics~Technology}                          {cgt}
  \ZZadd {Computer~Science}                                      {cs}
  \ZZadd {Construction~Management~Technology}                    {cm}
  \ZZadd {Consumer~Science}                                      {csr}
  \ZZadd {Curriculum~and~Instruction}                            {edci}
  \ZZadd {Earth,~Atmospheric,~and~Planetary~Sciences}            {eaps}
  \ZZadd {Economics}                                             {econ}
  \ZZadd {Educational~Studies}                                   {edst}
  \ZZadd {Electrical~and~Computer~Engineering}                   {ece}
  \ZZadd {Engineering~Education}                                 {ene}
  \ZZadd {Engineering~Technology}                                {et}
  \ZZadd {English}                                               {engl}
  \ZZadd {Entomology}                                            {entm}
  \ZZadd {Environmental~and~Ecological~Engineering}              {eee}
  \ZZadd {Food~Science}                                          {fs}
  \ZZadd {Forensic~and~Investigative~Sciences}                   {fis?}
  \ZZadd {Forestry~and~Natural~Resources}                        {fnr}
  \ZZadd {Health~Sciences}                                       {hsci}
  \ZZadd {Health~and~Kinesiology}                                {hk}
  \ZZadd {History}                                               {hist}
  \ZZadd {History~and~Philosophy}                                {histphil}
  \ZZadd {Horticulture}                                          {hort}
  \ZZadd {Hospitality~and~Tourism~Management}                    {htm}
  \ZZadd {Human~Development~and~Family~Science}                  {hdfs}
  \ZZadd {Industrial~and~Physical~Pharmacy}                      {ipph}
  \ZZadd {Industrial~Engineering}                                {ie}
  \ZZadd {Interdisciplinary~Biomedical~Studies~Program}          {ibsc}
  \ZZadd {Interdisciplinary~Studies~(Comparitive~Literature)}    {cmpl}
  \ZZadd {Languages~and~Cultures}                                {lc}
  \ZZadd {Linguistics}                                           {ling}
  \ZZadd {Management}                                            {mgmt}
  \ZZadd {Materials~Engineering}                                 {mse}
  \ZZadd {Mathematical~Sciences}                                 {math}
  \ZZadd {Mathematics}                                           {math}
  \ZZadd {Mechanical~Engineering}                                {me}
  \ZZadd {Mechnical~and~Civil~Engineering}                       {mce}
  \ZZadd {Medicinal~Chemistry~and~Molecular~Pharmacology}        {mcmp}
  \ZZadd {Music~Education}                                       {musice}
  \ZZadd {Music~Theory}                                          {musict}
  \ZZadd {Musicology}                                            {musico}
  \ZZadd {Nuclear~Engineering}                                   {ne}
  \ZZadd {Nursing}                                               {nur}
  \ZZadd {Nutrition~Science}                                     {nutr}
  \ZZadd {Organizational~Behavior~and~Human~Resource~Management} {obhr}
  \ZZadd {Pharmacy~Practice}                                     {phpr}
  \ZZadd {Philosophy}                                            {phil}
  \ZZadd {Philosophy~and~Literature}                             {phillit}
  \ZZadd {Physics}                                               {phys}
  \ZZadd {Physics~and~Astronomy}                                 {phys}
  \ZZadd {Political~Science}                                     {pol}
  \ZZadd {Psychological~Sciences}                                {psy}
  \ZZadd {PurdueThesis~Software~Development}                     {psd}
  \ZZadd {Sociology}                                             {soc}
  \ZZadd {Speech,~Language,~and~Hearing~Sciences}                {slhs}
  \ZZadd {Statistics}                                            {stat}
  \ZZadd {Technology}                                            {tech}
  \ZZadd {Technology~Leadership~and~Innovation}                  {tli}
  \ZZadd {Theatre}                                               {thtr}
  \ZZadd {Thesis~and~Dissertation~Office}                        {tdo}
  \ZZadd {Veterinary~Clinical~Sciences}                          {vcs}
  \ZZadd {Writing~Lab}                                           {owl}
  \ZZadd {Youth~Development~and~Agricultural~Education}          {ydae}
  \ZZcheck

  \ZZCheck:nnn {\A.+\Z} {ZZprogramhead} {one or more characters}

  \ZZCheck:nnn {\A.+\Z} {ZZtitle} {one or more characters}

  \ZZCheck:nnn {\A.+\Z} {ZZabbreviationsname} {one or more characters}

  \ZZCheck:nnn {\A.+\Z} {ZZabstractname} {one or more characters}

  \ZZCheck:nnn {\A.+\Z} {ZZacknowledgmentsname} {one or more characters}

  \ZZCheck:nnn {\A\d+(\.\d+)?\Z} {baselinestretch} {a real number}

  \ZZCheck:nnn {\A(true|false)\Z} {ZZchapterblankpages} {'true'~or~'false'}
  \bool_new:N \ZZchapterblankpagesB
  \tl_if_eq:NnTF \ZZchapterblankpages {true} {\bool_set_eq:NN \ZZchapterblankpagesB \c_true_bool} {}

  \ZZCheck:nnn {\A(true|false)\Z} {ZZshowcolophon} {'true'~or~'false'}
  \bool_new:N \ZZshowcolophonB
  \tl_if_eq:NnTF \ZZshowcolophon {true} {\bool_set_eq:NN \ZZshowcolophonB \c_true_bool} {}

  \ZZCheck:nnn {\A(true|false)\Z} {ZZcoversheets} {'true'~or~'false'}
  \bool_new:N \ZZcoversheetsB
  \tl_if_eq:NnTF \ZZcoversheets {true} {\bool_set_eq:NN \ZZcoversheetsB \c_true_bool} {}

  \ZZCheck:nnn {\A(true|false)\Z} {ZZshowdiagonalline} {'true'~or~'false'}
  \bool_new:N \ZZshowdiagonallineB
  \tl_if_eq:NnTF \ZZshowdiagonalline {true} {\bool_set_eq:NN \ZZshowdiagonallineB \c_true_bool} {}

  \ZZCheck:nnn {\A(true|false)\Z} {ZZendnotes} {'true'~or~'false'}
  \bool_new:N \ZZendnotesB
  \tl_if_eq:NnTF \ZZendnotes {true} {\bool_set_eq:NN \ZZendnotesB \c_true_bool} {}

  \ZZCheck:nnn {\A(true|false)\Z} {ZZfrenchspacing} {'true'~or~'false'}
  \bool_new:N \ZZfrenchspacingB
  \tl_if_eq:NnTF \ZZfrenchspacing {true} {\bool_set_eq:NN \ZZfrenchspacingB \c_true_bool} {}

  \ZZCheck:nnn {\A(true|false)\Z} {ZZshowgridlines} {'true'~or~'false'}
  \bool_new:N \ZZshowgridlinesB
  \tl_if_eq:NnTF \ZZshowgridlines {true} {\bool_set_eq:NN \ZZshowgridlinesB \c_true_bool} {}

  \ZZCheck:nnn {\A.+\Z} {ZZglossaryname} {one or more characters}

  \ZZCheck:nnn {\A(true|false)\Z} {ZZiso} {'true'~or~'false'}
  \bool_new:N \ZZisoB
  \tl_if_eq:NnTF \ZZiso {true} {\bool_set_eq:NN \ZZisoB \c_true_bool} {}

  \ZZCheck:nnn {\A.+\Z} {ZZlistoffiguresname} {one or more characters}

  \ZZCheck:nnn {\A.+\Z} {ZZlistoflistingsname} {one or more characters}

  \ZZCheck:nnn {\A.+\Z} {ZZlistofprotocolsname} {one or more characters}

  \ZZCheck:nnn {\A.+\Z} {ZZlistofschemesname} {one or more characters}

  \ZZCheck:nnn {\A.+\Z} {ZZlistoftablesname} {one or more characters}

  \ZZCheck:nnn {\A(true|false)\Z} {ZZshowmarginlines} {'true'~or~'false'}

  \ZZCheck:nnn {\A.+\Z} {ZZnomenclaturename} {one or more characters}

  \ZZCheck:nnn {\A(oneside|pdf|twoside)\Z} {ZZoutput} {'oneside'~or~'pdf'~or~'twoside'}

  \ZZCheck:nnn {\A.+\Z} {ZZprefacename} {one or more characters}

  \ZZCheck:nnn {\A.+\Z} {ZZsymbolsname} {one or more characters}

  \ZZCheck:nnn {\A.+\Z} {ZZtableofcontentsname} {one or more characters}

  \ZZCheck:nnn {\A(true|false)\Z} {ZZshowtimestamp} {'true'~or~'false'}

  \ZZCheck:nnn {\A(false|true|wide)\Z} {ZZshowtodonotes} {'false'~or~'true'~or~'wide'}

  \ZZCheck:nnn {\A.+\Z} {ZZvitaname} {one or more characters}

  \edef\ZZinscam{\ZZins-\ZZcam}
  \edef\ZZinscampro{\ZZins-\ZZcam-\ZZpro}
  \edef\ZZinscamprodeg{\ZZins-\ZZcam-\ZZpro-\ZZdeg}
  \typeout{}
  \typeout{OPTIONS}
  \typeout{}
  \typeout{MANDATORY▯PARAMETERS}
  \typeout{▯▯INPUT}
  \typeout{▯▯▯▯author▯▯▯▯▯▯▯▯▯▯'\ZZauthor'}
  \typeout{▯▯▯▯campus▯▯▯▯▯▯▯▯▯▯'\ZZcampus'}
  \typeout{▯▯▯▯degree▯▯▯▯▯▯▯▯▯▯'\ZZdegree'}
  \typeout{▯▯▯▯document▯▯▯▯▯▯▯▯'\ZZdocument'}
  \typeout{▯▯▯▯graduation▯▯▯▯▯▯'\ZZgraduation'}
  \typeout{▯▯▯▯institution▯▯▯▯▯'\ZZinstitution'}
  \typeout{▯▯▯▯program▯▯▯▯▯▯▯▯▯'\ZZprogram'}
  \typeout{▯▯SHORTENED}
  \typeout{▯▯▯▯ZZcam▯▯▯▯▯▯▯▯▯▯▯'\ZZcam'}
  \typeout{▯▯▯▯ZZdeg▯▯▯▯▯▯▯▯▯▯▯'\ZZdeg'}
  \typeout{▯▯▯▯ZZins▯▯▯▯▯▯▯▯▯▯▯'\ZZins'}
  \typeout{▯▯▯▯ZZpro▯▯▯▯▯▯▯▯▯▯▯'\ZZpro'}
  \typeout{▯▯DERIVED}
  \typeout{▯▯▯▯ZZinscam▯▯▯▯▯▯▯▯'\ZZinscam'}
  \typeout{▯▯▯▯ZZinscampro▯▯▯▯▯'\ZZinscampro'}
  \typeout{▯▯▯▯ZZinscamprodeg▯▯'\ZZinscamprodeg'}

  \seq_clear:N \l_tmpa_seq
  \seq_set_from_clist:Nn
    \l_tmpa_seq
    {
      uh-ma-econ-phd,
      iu-bl-musice-ma,    iu-bl-musice-mls,  iu-bl-musice-phd,
      iu-bl-musico-ma,    iu-bl-musico-mls,  iu-bl-musico-phd,
      iu-bl-musict-ma,    iu-bl-musict-mls,  iu-bl-musict-phd,
      unr-re-me-ms,       unr-re-me-phd,
      pu-fw-ap-ms,
      pu-fw-bio-ms,
      pu-fw-cme?-mse,
      pu-fw-comm-ma,      pu-fw-comm-ms,
      pu-fw-cs-ms,
      pu-fw-ece-mse,
      pu-fw-engl-ma,
      pu-ha-bio-ms,
      pu-ha-cit-ms,
      pu-ha-ece-ms,       pu-ha-ece-msece,
      pu-ha-hdfs-ms,
      pu-ha-hist-ma,
      pu-ha-mce-ms,       pu-ha-mce-mse,
      pu-in-bio-ms,
      pu-in-bme-msbme,    pu-in-bme-phd,
      pu-in-ccb?-ms,      pu-in-ccb?-phd,
      pu-in-cis?-ms,      pu-in-cis?-phd,
      pu-in-ece-ms,       pu-in-ece-mse,     pu-in-ece-msece,  pu-in-ece-phd,
      pu-in-fis?-ms,
      pu-in-math-ms,      pu-in-math-phd,
      pu-in-me-ms,        pu-in-me-mse,      pu-in-me-msme,    pu-in-me-phd,
      pu-in-mste-mse,
      pu-in-phys-ms,      pu-in-phys-phd,
      pu-we-bio-ms,
      pu-we-ece-ms,
      pu-we-hdfs-ms,
      pu-we-hist-ma,
      pu-we-mce-ms,       pu-we-mce-mse,
      pu-wl-aae-ms,       pu-wl-aae-msaa,    pu-wl-aae-phd,
      pu-wl-abe-ms,       pu-wl-abe-msabe,   pu-wl-abe-mse,    pu-wl-abe-phd,
      pu-wl-agecon-ms,    pu-wl-agecon-phd,
      pu-wl-agry-ms,      pu-wl-agry-phd,
      pu-wl-amst-ma,      pu-wl-amst-phd,
      pu-wl-ansc-ms,      pu-wl-ansc-phd,
      pu-wl-anth-ms,      pu-wl-anth-phd,
      pu-wl-ad-ma,        pu-wl-ad-mfa,      pu-wl-ad-phd,
      pu-wl-asec-ms,      pu-wl-asec-phd,
      pu-wl-satt-ms,      pu-wl-satt-msaam,  pu-wl-satt-phd,
      pu-wl-bms-ms,       pu-wl-bms-phd,
      pu-wl-bchm-ms,      pu-wl-bchm-phd,
      pu-wl-bme-msbme,    pu-wl-bme-phd,
      pu-wl-biol-ms,      pu-wl-biol-phd,
      pu-wl-btny-ms,      pu-wl-btny-phd,
      pu-wl-cce-ms,       pu-wl-cce-msce,    pu-wl-cce-mse,    pu-wl-cce-phd,
      pu-wl-chem-ms,      pu-wl-chem-phd,
      pu-wl-che-msche,    pu-wl-che-phd,
      pu-wl-cgt-ms,       pu-wl-cgt-phd,
      pu-wl-cit-ms,       pu-wl-cit-phd,
      pu-wl-com-ma,       pu-wl-com-ms,      pu-wl-com-phd,
      pu-wl-cpb-ms,       pu-wl-cpb-phd,
      pu-wl-cs-ms,        pu-wl-cs-phd,
      pu-wl-cm-ms,
      pu-wl-csr-ms,       pu-wl-csr-phd,
      pu-wl-edci-ms,      pu-wl-edci-msed,   pu-wl-edci-phd,
      pu-wl-eaps-ms,      pu-wl-eaps-phd,
      pu-wl-econ-ms,      pu-wl-econ-phd,
      pu-wl-edst-ms,      pu-wl-edst-msed,   pu-wl-edst-phd,
      pu-wl-eee-ms,       pu-wl-eee-phd,
      pu-wl-ece-ms,       pu-wl-ece-mse,     pu-wl-ece-msece,  pu-wl-ece-phd,
      pu-wl-ene-ms,       pu-wl-ene-mse,     pu-wl-ene-phd,
      pu-wl-engl-ma,      pu-wl-engl-mfa,    pu-wl-engl-phd,
      pu-wl-entm-ms,      pu-wl-entm-phd,
      pu-wl-et-mset,      pu-wl-et-phdtech,
      pu-wl-fs-ms,        pu-wl-fs-phd,
      pu-wl-fnr-ms,       pu-wl-fnr-phd,
      pu-wl-hk-ms,        pu-wl-hk-phd,
      pu-wl-hsci-ms,      pu-wl-hsci-phd,
      pu-wl-hdfs-ms,      pu-wl-hdfs-phd,
      pu-wl-hist-ma,      pu-wl-hist-phd,
      pu-wl-hort-ms,      pu-wl-hort-msa,    pu-wl-hort-phd,
      pu-wl-htm-ms,       pu-wl-htm-phd,
      pu-wl-ipph-ms,      pu-wl-ipph-phd,
      pu-wl-ie-ms,        pu-wl-ie-msie,     pu-wl-ie-phd,
      pu-wl-ibsc-phd,
      pu-wl-cmpl-ma,      pu-wl-cmpl-phd,
      pu-wl-lc-ma,        pu-wl-lc-phd,
      pu-wl-ling-ma,      pu-wl-ling-phd,
      pu-wl-mgmt-ms,      pu-wl-mgmt-phd,
      pu-wl-math-ms,      pu-wl-math-phd,
      pu-wl-mse-ms,       pu-wl-mse-msmse,   pu-wl-mse-phd,
      pu-wl-me-ms,        pu-wl-me-mse,      pu-wl-me-msme,    pu-wl-me-phd,
      pu-wl-mcmp-ms,      pu-wl-mcmp-phd,
      pu-wl-ne-msne,      pu-wl-ne-phd,
      pu-wl-nur-dnp,      pu-wl-nur-ms,      pu-wl-nur-phd,
      pu-wl-nutr-ms,      pu-wl-nutr-phd,
      pu-wl-obhr-mshrm,   pu-wl-obhr-phd,
      pu-wl-phil-ma,      pu-wl-phil-phd,
      pu-wl-phillit-phd,
      pu-wl-phpr-ms,      pu-wl-phpr-phd,
      pu-wl-phys-ms,      pu-wl-phys-phd,
      pu-wl-pol-ma,       pu-wl-pol-phd,
      pu-wl-psy-ms,       pu-wl-psy-phd,
      pu-wl-soc-ms,       pu-wl-soc-phd,
      pu-wl-slhs-aud,     pu-wl-slhs-ms,     pu-wl-slhs-phd,
      pu-wl-stat-ms,      pu-wl-stat-phd,
      pu-wl-tech-dtech,   pu-wl-tech-phd,
      pu-wl-tli-ms,       pu-wl-tli-msit,    pu-wl-tli-phd,
      pu-wl-thtr-mfa,
      pu-wl-vcs-ms,       pu-wl-vcs-phd,
      pu-wl-ydae-ms,      pu-wl-ydae-phd
    }
  % For debugging.
  % \seq_show:N \l_tmpa_seq

  \prop_clear:N \l_tmpa_prop
  \seq_map_inline:Nn
    {\l_tmpa_seq}
    {
      \prop_put:Nnn \l_tmpa_prop {#1} {1}
    }
  % For debugging.
  % \prop_show:N \l_tmpa_prop

  % \seq_sort:Nn
  %   {\l_tmpa_seq}
  %   {
  %     \int_compare:nNnTF { #1 } > { #2 }
  %       { \sort_return_swapped: }
  %       { \sort_return_same: }
  %   }

  \prop_get:NVNTF
    \l_tmpa_prop
    \ZZinscamprodeg
    \ZZtemplate
    % TRUE CODE
    {
      \typeout{}
      \typeout{(}
      \typeout{▯▯\ZZDebuggingMessage}
      \typeout{▯▯This▯institution/campus/program/degree▯combination▯is▯ok:}
      \typeout{▯▯▯▯\ZZbs ZZinstitution▯is▯'\ZZinstitution'}
      \typeout{▯▯▯▯\ZZbs ZZcampus▯▯▯▯▯▯is▯'\ZZcampus'}
      \typeout{▯▯▯▯\ZZbs ZZprogram▯▯▯▯▯is▯'\ZZprogram'}
      \typeout{▯▯▯▯\ZZbs ZZdegree▯▯▯▯▯▯is▯'\ZZdegree'}
      \typeout{)}
    }
    % FALSE CODE
    {
      \seq_sort:Nn
        {\l_tmpa_seq}
        {
          % Split into ins, cam, pro, and deg.
          \seq_set_split:Nnn \l_a_seq {-} {#1}
          \seq_set_split:Nnn \l_b_seq {-} {#2}
          \typeout{comparing▯#1▯and▯#2} %%%%
    
          % Process ins.
          \seq_pop_left:NN \l_a_seq \l_tmpa_tl
          \seq_pop_left:NN \l_b_seq \l_tmpb_tl
          \typeout{▯▯▯▯comparing▯\l_tmpa_tl▯and▯\l_tmpb_tl} %%%%
          \prop_get:NnN {\l_ins_prop} {\l_tmpa_tl} {\l_tmpa_tl}
          \prop_get:NnN {\l_ins_prop} {\l_tmpb_tl} {\l_tmpb_tl}
          \str_compare:nNnTF {\l_tmpa_tl} > {\l_tmpb_tl}
            { \sort_return_swapped: }
            { }
    
          % Process cam.
          \seq_pop_left:NN \l_a_seq \l_tmpa_tl
          \seq_pop_left:NN \l_b_seq \l_tmpb_tl
          \typeout{▯▯▯▯▯▯▯▯comparing▯\l_tmpa_tl▯and▯\l_tmpb_tl} %%%%
          \prop_get:NnN {\l_cam_prop} {\l_tmpa_tl} {\l_tmpa_tl}
          \prop_get:NnN {\l_cam_prop} {\l_tmpb_tl} {\l_tmpb_tl}
          \str_compare:nNnTF {\l_tmpa_tl} > {\l_tmpb_tl}
            { \sort_return_swapped: }
            { }
    
          % Process pro.
          \seq_pop_left:NN \l_a_seq \l_tmpa_tl
          \seq_pop_left:NN \l_b_seq \l_tmpb_tl
          \typeout{▯▯▯▯▯▯▯▯▯▯▯▯comparing▯\l_tmpa_tl▯and▯\l_tmpb_tl} %%%%
          \prop_get:NnN {\l_pro_prop} {\l_tmpa_tl} {\l_tmpa_tl}
          \prop_get:NnN {\l_pro_prop} {\l_tmpb_tl} {\l_tmpb_tl}
          \str_compare:nNnTF {\l_tmpa_tl} > {\l_tmpb_tl}
            { \sort_return_swapped: }
            { }
    
          % Process deg.
          \seq_pop_left:NN \l_a_seq \l_tmpa_tl
          \seq_pop_left:NN \l_b_seq \l_tmpb_tl
          \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯comparing▯\l_tmpa_tl▯and▯\l_tmpb_tl} %%%%
          \prop_get:NnN {\l_deg_prop} {\l_tmpa_tl} {\l_tmpa_tl}
          \prop_get:NnN {\l_deg_prop} {\l_tmpb_tl} {\l_tmpb_tl}
          \str_compare:nNnTF {\l_tmpa_tl} > {\l_tmpb_tl}
            { \sort_return_swapped: }
            { \sort_return_same: }
        }
    
      \typeout{}
      \typeout{THIS▯IS▯A▯BAD▯INSTITUTION/CAMPUS/PROGRAM/DEGREE▯COMBINATION:}
      \typeout{}

      \typeout{▯▯▯▯\ZZbs ZZinstitution▯is▯'\ZZinstitution'}
      \typeout{▯▯▯▯\ZZbs ZZcampus▯▯▯▯▯▯is▯'\ZZcampus'}
      \typeout{▯▯▯▯\ZZbs ZZprogram▯▯▯▯▯is▯'\ZZprogram'}
      \typeout{▯▯▯▯\ZZbs ZZdegree▯▯▯▯▯▯is▯'\ZZdegree'}

      \typeout{}
      \typeout{THE▯LEGAL▯INSTITUTION/CAMPUS/PROGRAM/DEGREE▯COMBINATIONS▯ARE:}
      \typeout{}

%\tl_new:N {\l_insa_tl} \tl_new:N {\l_insb_tl}
%\tl_new:N {\l_cama_tl} \tl_new:N {\l_camb_tl}
%\tl_new:N {\l_proa_tl} \tl_new:N {\l_prob_tl}
%\tl_new:N {\l_dega_tl} \tl_new:N {\l_degb_tl}
%
%      \seq_map_inline:Nn
%        {\l_tmpa_seq}
%        {
%          % Split into ins, cam, pro, and deg.
%          \seq_set_split:Nnn \l_a_seq {-} {#1}
%          \seq_pop_left:NN \l_a_seq \l_insa_tl
%          \seq_pop_left:NN \l_a_seq \l_cama_tl
%          \seq_pop_left:NN \l_a_seq \l_proa_tl
%          \seq_pop_left:NN \l_a_seq \l_dega_tl
%          \typeout{▯▯▯▯\l_insa_tl ▯▯▯▯\l_cama_tl ▯▯▯▯\l_proa_tl ▯▯▯▯\l_dega_tl}
%          % Convert into institution, campus, program, and degree.
%          \prop_get:NnN \l_ins_prop \l_insa_tl \l_insb_tl
%          \tl_show:N \l_insb_tl
%          \typeout{1111}
%          \prop_get:NnN {\l_cam_prop} {\l_cama_tl} {\l_camb_tl}
%          \typeout{2222}
%          \prop_get:NnN {\l_pro_prop} {\l_proa_tl} {\l_prob_tl}
%          \typeout{3333}
%          \prop_get:NnN {\l_deg_prop} {\l_dega_tl} {\l_degb_tl}
%          \typeout{4444}
%          \typeout{▯▯▯▯5555}
%        }
%\ZZAbort


\typeout{▯▯▯▯University▯of▯Hawaii}
      \typeout{▯▯▯▯▯▯▯▯Manoa}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Economics}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Indiana▯University}
      \typeout{▯▯▯▯▯▯▯▯Bloomington}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Music▯Education}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Arts}
      \typeout{}
      \typeout{▯▯▯▯Indiana▯University}
      \typeout{▯▯▯▯▯▯▯▯Bloomington}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Music▯Education}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Library▯Science}
      \typeout{}
      \typeout{▯▯▯▯Indiana▯University}
      \typeout{▯▯▯▯▯▯▯▯Bloomington}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Music▯Education}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Indiana▯University}
      \typeout{▯▯▯▯▯▯▯▯Bloomington}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Music▯Theory}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Arts}
      \typeout{}
      \typeout{▯▯▯▯Indiana▯University}
      \typeout{▯▯▯▯▯▯▯▯Bloomington}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Music▯Theory}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Library▯Science}
      \typeout{}
      \typeout{▯▯▯▯Indiana▯University}
      \typeout{▯▯▯▯▯▯▯▯Bloomington}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Music▯Theory}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Indiana▯University}
      \typeout{▯▯▯▯▯▯▯▯Bloomington}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Musicology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Arts}
      \typeout{}
      \typeout{▯▯▯▯Indiana▯University}
      \typeout{▯▯▯▯▯▯▯▯Bloomington}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Musicology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Library▯Science}
      \typeout{}
      \typeout{▯▯▯▯Indiana▯University}
      \typeout{▯▯▯▯▯▯▯▯Bloomington}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Musicology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯?????}
      \typeout{▯▯▯▯▯▯▯▯Reno}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Mechanical▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯?????}
      \typeout{▯▯▯▯▯▯▯▯Reno}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Mechanical▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Fort▯Wayne}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Applied▯Physics}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Fort▯Wayne}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Biological▯Sciences}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Fort▯Wayne}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Civil▯and▯Mechanical▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Engineering}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Fort▯Wayne}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Communication}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Arts}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Fort▯Wayne}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Communication}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Fort▯Wayne}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Computer▯Science}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Fort▯Wayne}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Electrical▯and▯Computer▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Engineering}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Fort▯Wayne}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯English}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Arts}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Hammond}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Biology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Hammond}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Human▯Development▯and▯Family▯Science}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Hammond}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Electrical▯and▯Computer▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Hammond}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Electrical▯and▯Computer▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Electrical▯and▯Computer▯Engineering}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Hammond}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯History▯and▯Philosphy}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Arts}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Hammond}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Mechnical▯and▯Civil▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Hammond}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Mechnical▯and▯Civil▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Engineering}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Indianapolis}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Biology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Indianapolis}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Biomedical▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Indianapolis}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Biomedical▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Indianapolis}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Chemistry▯and▯Chemical▯Biology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Indianapolis}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Chemistry▯and▯Chemical▯Biology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Indianapolis}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Computer▯and▯Information▯Science}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Indianapolis}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Computer▯and▯Information▯Science}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Indianapolis}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Electrical▯and▯Computer▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Indianapolis}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Electrical▯and▯Computer▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Engineering}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Indianapolis}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Electrical▯and▯Computer▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Indianapolis}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Forensic▯and▯Investigative▯Sciences}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Indianapolis}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Mathematical▯Sciences}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Indianapolis}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Mathematical▯Sciences}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Indianapolis}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Mechanical▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Indianapolis}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Mechanical▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Engineering}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Indianapolis}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Mechanical▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Mechanical▯Engineering}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Indianapolis}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Mechanical▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Indianapolis}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Physics}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Indianapolis}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Physics}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Westville}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Biology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Westville}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Electrical▯and▯Computer▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Westville}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯History▯and▯Philosphy}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Arts}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Westville}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Human▯Development▯and▯Family▯Science}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Westville}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Mechnical▯and▯Civil▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯Westville}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Mechnical▯and▯Civil▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Engineering}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Aeronautics▯and▯Astronautics}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Aeronautics▯and▯Astronautics}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Aeronautics▯and▯Astronautics}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Aeronautics▯and▯Astronautics}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Agricultural▯and▯Biological▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Agricultural▯and▯Biological▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Agricultural▯and▯Biological▯Engineering}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Agricultural▯and▯Biological▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Engineering}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Agricultural▯and▯Biological▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Agricultural▯Economics}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Agricultural▯Economics}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Agricultural Sciences Education and Communication}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Agricultural Sciences Education and Communication}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Agronomy}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Agronomy}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯American▯Studies}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Arts}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯American▯Studies}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Animal▯Sciences}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Animal▯Sciences}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Anthropology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Anthropology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Art▯and▯Design}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Arts}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Art▯and▯Design}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Fine▯Arts}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Art▯and▯Design}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Aviation▯and▯Transportation▯Technology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Aviation▯and▯Transportation▯Technology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Aviation▯and▯Aerospace▯Management}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Aviation▯and▯Transportation▯Technology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Basic▯Medical▯Sciences}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Basic▯Medical▯Sciences}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Biochemistry}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Biochemistry}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Biomedical▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Biomedical▯Engineering}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Biomedical▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Biological▯Sciences}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Biological▯Sciences}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Botany▯and▯Plant▯Pathology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Botany▯and▯Plant▯Pathology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Chemistry}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Chemistry}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Chemical▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Chemical▯Engineering}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Chemical▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Civil▯and▯Construction▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Civil▯and▯Construction▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Civil▯Engineering}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Civil▯and▯Construction▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Engineering}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Civil▯and▯Construction▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Communications}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Arts}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Communications}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Communications}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Comparitive▯Pathobiology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Comparitive▯Pathobiology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Computer▯and▯Information▯Technology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Computer▯Graphics▯Technology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Computer▯Graphics▯Technology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Computer▯Science}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Computer▯Science}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Construction▯Management▯Technology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Consumer▯Science}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Consumer▯Science}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Curriculum▯and▯Instruction}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Curriculum▯and▯Instruction}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Education}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Curriculum▯and▯Instruction}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Earth,▯Atmospheric,▯and▯Planetary▯Sciences}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Earth,▯Atmospheric,▯and▯Planetary▯Sciences}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Economics}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Economics}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Educational▯Studies}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Educational▯Studies}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Education}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Educational▯Studies}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Environmental▯and▯Ecological▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Environmental▯and▯Ecological▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Electrical▯and▯Computer▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Electrical▯and▯Computer▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Electrical▯and▯Computer▯Engineering}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Electrical▯and▯Computer▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Engineering}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Electrical▯and▯Computer▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Engineering▯Education}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Engineering▯Education}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Engineering}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Engineering▯Education}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Engineering▯Education}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Engineering▯Technology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Technology}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Engineering▯Technology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Ph.D.▯in▯Technology}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯English}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Arts}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯English}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Fine▯Arts}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯English}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Entomology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Entomology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Food▯Science}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Food▯Science}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Forestry▯and▯Natural▯Resources}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Forestry▯and▯Natural▯Resources}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Health▯and▯Kinesiology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Health▯and▯Kinesiology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Health▯Sciences}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Health▯Sciences}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Human▯Development▯and▯Family▯Science}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Human▯Development▯and▯Family▯Science}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯History}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Arts}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯History}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Horticulture}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Horticulture}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Agriculture}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Horticulture}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Hospitality▯and▯Tourism▯Management}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Hospitality▯and▯Tourism▯Management}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Industrial▯and▯Physical▯Pharmacy}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Industrial▯and▯Physical▯Pharmacy}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Industrial▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Industrial▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Industrial▯Engineering}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Industrial▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Interdisciplinary▯Biomedical▯Studies▯Program}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Interdisciplinary▯Studies▯(Comparitive▯Literature)}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Arts}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Interdisciplinary▯Studies▯(Comparitive▯Literature)}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Languages▯and▯Cultures}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Arts}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Languages▯and▯Cultures}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Linguistics}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Arts}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Linguistics}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Management}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Management}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Mathematics}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Mathematics}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Materials▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Materials▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Materials▯Engineering}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Materials▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Mechanical▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Mechanical▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Engineering}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Mechanical▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Mechanical▯Engineering}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Mechanical▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Medicinal▯Chemistry▯and▯Molecular▯Pharmacology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Medicinal▯Chemistry▯and▯Molecular▯Pharmacology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Nuclear▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Nuclear▯Engineering}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Nuclear▯Engineering}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Nursing}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Nursing▯Practice}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Nursing}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Nursing}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Nutrition▯Science}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Nutrition▯Science}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Organizational▯Behavior▯and▯Human▯Resource▯Management}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Human▯Resources▯Management}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Organizational▯Behavior▯and▯Human▯Resource▯Management}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Philosophy}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Arts}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Philosophy}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Philosophy▯and▯Literature}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Pharmacy▯Practice}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Pharmacy▯Practice}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Physics▯and▯Astronomy}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Physics▯and▯Astronomy}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Political▯Science}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Arts}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Political▯Science}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Psychological▯Sciences}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Psychological▯Sciences}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Sociology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Sociology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Speech,▯Language,▯and▯Hearing▯Sciences}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Audiology}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Speech,▯Language,▯and▯Hearing▯Sciences}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Speech,▯Language,▯and▯Hearing▯Sciences}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Statistics}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Statistics}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Technology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Technology}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Technology}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Technology▯Leadership▯and▯Innovation}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Technology▯Leadership▯and▯Innovation}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science▯in▯Industrial▯Technology}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Technology▯Leadership▯and▯Innovation}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Theatre}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Fine▯Arts}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Veterinary▯Clinical▯Sciences}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Veterinary▯Clinical▯Sciences}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Youth▯Development▯and▯Agricultural▯Education}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Master▯of▯Science}
      \typeout{}
      \typeout{▯▯▯▯Purdue▯University}
      \typeout{▯▯▯▯▯▯▯▯West▯Lafayette}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯Youth▯Development▯and▯Agricultural▯Education}
      \typeout{▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯Doctor▯of▯Philosophy}
      \typeout{}
      \ZZAbort
    }

  \bool_new:N \ZZCenterCaptionB
  \bool_set_true:N \ZZCenterCaptionB

  \bool_new:N \ZZChapterBlankPagesB
  \bool_set_false:N \ZZChapterBlankPagesB

  \bool_new:N \ZZCoverSheetsB
  \bool_set_false:N \ZZCoverSheetsB

  \bool_new:N \ZZFoundB
  \bool_set_false:N \ZZFoundB

  \bool_new:N \ZZinappendixb
  \bool_set_false:N \ZZinappendixb

  \bool_new:N \ZZInChaptersB
  \bool_set_false:N \ZZInChaptersB

  \bool_new:N \ZZinotherb
  \bool_set_false:N \ZZinotherb

  \bool_new:N \ZZNonChapterBlankPagesB
  \bool_set_false:N \ZZNonChapterBlankPagesB

  \bool_new:N \ZZNumberAllVolumesB
  \bool_set_false:N \ZZNumberAllVolumesB

  \bool_new:N \ZZUnsetB
  \bool_set_false:N \ZZUnsetB

  \setcounter{ZZprevindex}{0}
  \bool_new:N \ZZtitlepresent
  \newcommand{\ZZtitlemessage}{\string \maketitle}
  \newcounter{ZZtitlepos}   \setcounter{ZZtitlepos}{1}
  \newcounter{ZZtitleincr}  \setcounter{ZZtitleincr}{1}

  \bool_new:N \ZZdedicationpresent
  \newcommand{\ZZdedicationmessage}{\string dedication}
  \newcounter{ZZdedicationpos}   \setcounter{ZZdedicationpos}{2}
  \newcounter{ZZdedicationincr}  \setcounter{ZZdedicationincr}{1}

  \bool_new:N \ZZacknowledgmentspresent
  \newcommand{\ZZacknowledgmentsmessage}{\string acknowledgments}
  \newcounter{ZZacknowledgmentspos}   \setcounter{ZZacknowledgmentspos}{3}
  \newcounter{ZZacknowledgmentsincr}  \setcounter{ZZacknowledgmentsincr}{1}

  \bool_new:N \ZZprefacepresent
  \newcommand{\ZZprefacemessage}{\string preface}
  \newcounter{ZZprefacepos}   \setcounter{ZZprefacepos}{4}
  \newcounter{ZZprefaceincr}  \setcounter{ZZprefaceincr}{1}

  \bool_new:N \ZZtableofcontentspresent
  \newcommand{\ZZtableofcontentsmessage}{\string \tableofcontents}
  \newcounter{ZZtableofcontentspos}   \setcounter{ZZtableofcontentspos}{5}
  \newcounter{ZZtableofcontentsincr}  \setcounter{ZZtableofcontentsincr}{1}

  \bool_new:N \ZZlistoftablespresent
  \newcommand{\ZZlistoftablesmessage}{\string \listoftables}
  \newcounter{ZZlistoftablespos}   \setcounter{ZZlistoftablespos}{6}
  \newcounter{ZZlistoftablesincr}  \setcounter{ZZlistoftablesincr}{1}

  \bool_new:N \ZZlistoffigurespresent
  \newcommand{\ZZlistoffiguresmessage}{\string \listoffigures}
  \newcounter{ZZlistoffigurespos}   \setcounter{ZZlistoffigurespos}{7}
  \newcounter{ZZlistoffiguresincr}  \setcounter{ZZlistoffiguresincr}{1}

  \bool_new:N \ZZZZlistoflistingspresent
  \newcommand{\ZZZZlistoflistingsmessage}{\string \ZZlistoflistings}
  \newcounter{ZZZZlistoflistingspos}   \setcounter{ZZZZlistoflistingspos}{8}
  \newcounter{ZZZZlistoflistingsincr}  \setcounter{ZZZZlistoflistingsincr}{1}

  \bool_new:N \ZZlistofprotocolspresent
  \newcommand{\ZZlistofprotocolsmessage}{\string \ZZlistofprotocols}
  \newcounter{ZZlistofprotocolspos}   \setcounter{ZZlistofprotocolspos}{9}
  \newcounter{ZZlistofprotocolsincr}  \setcounter{ZZlistofprotocolsincr}{1}

  \bool_new:N \ZZlistofschemespresent
  \newcommand{\ZZlistofschemesmessage}{\string \ZZlistofschemes}
  \newcounter{ZZlistofschemespos}   \setcounter{ZZlistofschemespos}{10}
  \newcounter{ZZlistofschemesincr}  \setcounter{ZZlistofschemesincr}{1}

  \bool_new:N \ZZsymbolspresent
  \newcommand{\ZZsymbolsmessage}{\string symbols}
  \newcounter{ZZsymbolspos}   \setcounter{ZZsymbolspos}{11}
  \newcounter{ZZsymbolsincr}  \setcounter{ZZsymbolsincr}{1}

  \bool_new:N \ZZabbreviationspresent
  \newcommand{\ZZabbreviationsmessage}{\string abbreviations}
  \newcounter{ZZabbreviationspos}   \setcounter{ZZabbreviationspos}{12}
  \newcounter{ZZabbreviationsincr}  \setcounter{ZZabbreviationsincr}{1}

  \bool_new:N \ZZnomenclaturepresent
  \newcommand{\ZZnomenclaturemessage}{\string nomenclature}
  \newcounter{ZZnomenclaturepos}   \setcounter{ZZnomenclaturepos}{13}
  \newcounter{ZZnomenclatureincr}  \setcounter{ZZnomenclatureincr}{1}

  \bool_new:N \ZZglossarypresent
  \newcommand{\ZZglossarymessage}{\string glossary}
  \newcounter{ZZglossarypos}   \setcounter{ZZglossarypos}{14}
  \newcounter{ZZglossaryincr}  \setcounter{ZZglossaryincr}{1}

  \bool_new:N \ZZabstractpresent
  \newcommand{\ZZabstractmessage}{\string abstract}
  \newcounter{ZZabstractpos}   \setcounter{ZZabstractpos}{15}
  \newcounter{ZZabstractincr}  \setcounter{ZZabstractincr}{1}

  \bool_new:N \ZZchapterpresent
  \newcommand{\ZZchaptermessage}{\string chapter}
  \newcounter{ZZchapterpos}   \setcounter{ZZchapterpos}{16}
  \newcounter{ZZchapterincr}  \setcounter{ZZchapterincr}{0}

  \bool_new:N \ZZbibliographypresent
  \newcommand{\ZZbibliographymessage}{\string bibliography}
  \newcounter{ZZbibliographypos}   \setcounter{ZZbibliographypos}{17}
  \newcounter{ZZbibliographyincr}  \setcounter{ZZbibliographyincr}{1}

  \bool_new:N \ZZappendixpresent
  \newcommand{\ZZappendixmessage}{\string appendix}
  \newcounter{ZZappendixpos}   \setcounter{ZZappendixpos}{18}
  \newcounter{ZZappendixincr}  \setcounter{ZZappendixincr}{1}

  \bool_new:N \ZZappendicespresent
  \newcommand{\ZZappendicesmessage}{\string appendices}
  \newcounter{ZZappendicespos}   \setcounter{ZZappendicespos}{19}
  \newcounter{ZZappendicesincr}  \setcounter{ZZappendicesincr}{1}

  \bool_new:N \ZZachapterpresent
  \newcommand{\ZZachaptermessage}{\string chapter}
  \newcounter{ZZachapterpos}   \setcounter{ZZachapterpos}{20}
  \newcounter{ZZachapterincr}  \setcounter{ZZachapterincr}{0}

  \bool_new:N \ZZnotesandfootnotespresent
  \newcommand{\ZZnotesandfootnotesmessage}{\string notesandfootnotes}
  \newcounter{ZZnotesandfootnotespos}   \setcounter{ZZnotesandfootnotespos}{21}
  \newcounter{ZZnotesandfootnotesincr}  \setcounter{ZZnotesandfootnotesincr}{1}

  \bool_new:N \ZZvitapresent
  \newcommand{\ZZvitamessage}{\string vita}
  \newcounter{ZZvitapos}   \setcounter{ZZvitapos}{22}
  \newcounter{ZZvitaincr}  \setcounter{ZZvitaincr}{1}

  \bool_new:N \ZZpublicationpresent
  \newcommand{\ZZpublicationmessage}{\string publication}
  \newcounter{ZZpublicationpos}   \setcounter{ZZpublicationpos}{23}
  \newcounter{ZZpublicationincr}  \setcounter{ZZpublicationincr}{1}

  \bool_new:N \ZZpublicationspresent
  \newcommand{\ZZpublicationsmessage}{\string publications}
  \newcounter{ZZpublicationspos}   \setcounter{ZZpublicationspos}{24}
  \newcounter{ZZpublicationsincr}  \setcounter{ZZpublicationsincr}{1}

  \newcommand{\ZZpresentandinorder}[1]
    {
      \bool_set_true:c {ZZ#1present}
      \setcounter{ZZnameindex}{\arabic{ZZ#1pos}}
      \ifthenelse{\value{ZZnameindex} < \value{ZZprevindex}}
        {
          \typeout{}
          \typeout{}
          \typeout{The▯"#1"▯part▯of▯the▯document▯is▯out▯of▯order.}
          \typeout{The▯existing▯parts▯of▯the▯document▯should▯be▯in▯this▯order:}
          \typeout{▯▯▯▯\ZZbs maketitle}
          \typeout{▯▯▯▯\ZZbs begin{dedication}▯▯▯▯▯▯▯▯...▯\ZZbs end{dedication}}
          \typeout{▯▯▯▯\ZZbs begin{acknowledgments}▯▯▯...▯\ZZbs end{acknowledgments}}
          \typeout{▯▯▯▯\ZZbs begin{preface}▯▯▯▯▯▯▯▯▯▯▯...▯\ZZbs end{preface}}
          \typeout{▯▯▯▯\ZZbs tableofcontents}
          \typeout{▯▯▯▯\ZZbs listoftables}
          \typeout{▯▯▯▯\ZZbs listoffigures}
          \typeout{▯▯▯▯\ZZbs ZZlistofprotocols}
          \typeout{▯▯▯▯\ZZbs ZZlistofschemes}
          \typeout{▯▯▯▯\ZZbs begin{symbols}▯▯▯▯▯▯▯▯▯▯▯...▯\ZZbs end{symbols}}
          \typeout{▯▯▯▯\ZZbs begin{abbreviations}▯▯▯▯▯...▯\ZZbs end{abbreviations}}
          \typeout{▯▯▯▯\ZZbs begin{nomenclature}▯▯▯▯▯▯...▯\ZZbs end{nomenclature}}
          \typeout{▯▯▯▯\ZZbs begin{glossary}▯▯▯▯▯▯▯▯▯▯...▯\ZZbs end{glossary}}
          \typeout{▯▯▯▯\ZZbs begin{abstract}▯▯▯▯▯▯▯▯▯▯...▯\ZZbs end{abstract}}
          \typeout{▯▯▯▯\ZZbs chapter}
          \typeout{▯▯▯▯bibliography}
          \typeout{▯▯▯▯\ZZbs appendix▯▯or▯▯\ZZbs appendices}
          \typeout{▯▯▯▯\ZZbs chapter}
          \typeout{▯▯▯▯\ZZbs begin{notesandfootnotes}▯...▯\ZZbs end{notesandfootnotes}}
          \typeout{▯▯▯▯\ZZbs begin{vita}▯▯▯▯▯▯▯▯▯▯▯▯▯▯...▯\ZZbs end{vita}}
          \typeout{▯▯▯▯▯▯▯▯\ZZbs begin{publication}▯▯▯...▯\ZZbs end{publication}}
          \typeout{▯▯▯▯or▯▯\ZZbs begin{publications}▯▯...▯\ZZbs end{publications}}
          \see_error_immediately_above
        }
        {\setcounter{ZZprevindex}{\value{ZZnameindex}}}
    }

  \newcommand{\bcaption}
    {
      \bool_set_true:N \ZZCenterCaptionB
      \refstepcounter\@captype \@dblarg{\@caption\@captype}
    }

  \DeclareRobustCommand{\okina}
    {
      \raisebox
        {\dimexpr\fontcharht\font`A-\height}
        {\scalebox{0.8}{`}}
    }

  \newcommand{\ZZnottableofcontents}{}

  \NeedsTeXFormat{LaTeX2e}

  \ifpdf
  \else
    \typeout{This▯document▯must▯be▯processed▯using▯pdflatex}
    \typeout{so▯figures▯are▯processed▯correctly.}
    \ZZAbort
  \fi

  \clubpenalty         = 10000
  \displaywidowpenalty = 10000
  \widowpenalty        = 10000

  \def\@dottedtocline#1#2#3#4#5#6
    {
      \ifnum #1>\c@tocdepth
      \else
        \vskip #2
        {
          \leftskip #3
          \rightskip \@tocrmarg
          \parfillskip -\rightskip
          \parindent = #3
          \@afterindenttrue
          \interlinepenalty\@M
          \leavevmode
          \@tempdima = #4
          \advance\@tempdima \ZZpaddinglength
          \advance\leftskip \@tempdima
          \hbox{}\hskip -\leftskip
          #5\nobreak
          \leaders\hbox{$\m@th \mkern \@dotsep mu.\mkern \@dotsep mu$}\hfill
          \nobreak
          \renewcommand{\@pnumwidth}{1.55em}
          \setlength{\ZZtl}{\@pnumwidth}
          \settowidth{\ZZtlb}{\reset@font \rm #6}
          \ifdim \ZZtlb>\ZZtl
            \hbox to\ZZtlb{\hfil\reset@font \rm #6}\par
          \else
            \hbox to\ZZtl{\hfil\reset@font \rm #6}\par
          \fi
        }
      \fi
    }

  \def\big@toc@skip{0.5\baselineskip plus 0.2\p@}

  \def\small@toc@skip{-4pt}

  \newcommand{\l@protocol}{SET}
  \newcommand{\l@scheme}{SET}

  \ifthen{\equal{\ZZshowdiagonalline}{true} \or \equal{\ZZshowgridlines}{true} \or \equal{\ZZshowmarginlines}{true}}                                       
    {
      \usepackage{wallpaper}
      \ifthen{\equal{\ZZshowdiagonalline}{true}}
        {\ULCornerWallPaper{1}{gr-diagonalline.pdf}}
      \ifthen{\equal{\ZZshowgridlines}{true}}
        {\ULCornerWallPaper{1}{gr-gridlines.pdf}}
      \ifthen{\equal{\ZZshowmarginlines}{true}}
        {\ULCornerWallPaper{1}{gr-marginlines.pdf}}
    }

    \usepackage[calc,en-US,useregional]{datetime2}
      \DTMsavenow{ttime}
      \newcommand{\ZZDateRun}{\DTMfetchyear{ttime}-\DTMfetchmonth{ttime}-\DTMfetchday{ttime}}
      \newcommand{\ZZTimeRun}{\DTMfetchhour{ttime}:\DTMfetchminute{ttime}:\DTMfetchsecond{ttime}}
      \ifthen{\equal{\ZZshowtimestamp}{true}}
        {
          \newcommand{\MyDateDayTimeStamp}
            {
              compiled~on~\ZZDateRun\quad
              \DTMenglishweekdayname{\DTMfetchdow{ttime}}\quad
              \ZZTimeRun
            }
          \def\@evenhead{\hfil\tt\MyDateDayTimeStamp}
          \def\@oddhead{\hfil\tt\MyDateDayTimeStamp}
        }
      \str_case:VnTF
        {\ZZshowtodonotes}
        {
          {false}
            {
              \usepackage[letterpaper,left=1in,top=1.1in,bottom=1in,right=1in]{geometry}
              \usepackage[disable]{todonotes}
            }
          {true}
            {
              \usepackage[letterpaper,left=1in,top=1.1in,bottom=1in,right=1in]{geometry}
              \usepackage[color=white,textwidth=0.8in]{todonotes}
            }
          {wide}
            {
              \usepackage[papersize={11.5in,11in},left=1in,top=1.1in,bottom=1in,right=4in]{geometry}
              % See
              %   https://tug.ctan.org/macros/latex/contrib/todonotes/todonotes.pdf
              \usepackage[color=white,textwidth=2.8in]{todonotes}
                \setlength{\marginparwidth}{2.8in}
            }
        }
        % TRUE CODE
        { }
        % FALSE CODE
        { }

  \newcommand{\todocomment}[1]{\renewcommand{\baselinestretch}{1}\todo[color=green!90,tickmarkheight=6pt]{\small #1}}
  \newcommand{\todowarn}[1]{\renewcommand{\baselinestretch}{1}\todo[color=yellow!90,tickmarkheight=6pt]{\small #1}}
  \newcommand{\todoerror}[1]{\renewcommand{\baselinestretch}{1}\todo[color=red!55,tickmarkheight=6pt]{\small #1}}
  % Originally I was going to use gray for index entries, but the
  % lines to where they occur in the text didn't show up well enough.
  \newcommand{\todoindex}[1]{\renewcommand{\baselinestretch}{1}\todo[color=white,linecolor=black,tickmarkheight=6pt]{\small #1}}
  \seq_new:N \l_index_seq
  \def\ix#1
    {
      \seq_set_split:Nnn \l_index_seq {//} {#1}
      \def\ttt{\seq_use:Nn \l_index_seq {\penalty-100\kern0.3ex $\bullet$\nobreak\kern0.3ex}}
      \todoindex{\ttt}
      % This is a comment.
      \seq_map_inline:Nn \l_index_seq
      {
        \index{##1}
      }
    }

  \newcommand{\ZZbaselinestretch}[1]
    {
      \renewcommand{\baselinestretch}{#1}
      \reset@font
    }

  \newcommand{\ZZBlankPage}
    {
      \clearpage
      \mbox{}
      \clearpage
    }

  \AtBeginDocument
    {
      \usepackage{notoccite}
      \renewcommand{\@evenfoot}{\hfil\textrm{\thepage}\hfil}
      \renewcommand{\@oddfoot}{\hfil\textrm{\thepage}\hfil}
    }

  \newcommand{\ZZtitlefont}{\large\bf}
  \newcommand{\ZZauthorfont}{\bf}
  \newcommand{\ZZdocumentfont}{\bf}
  \newcommand{\ZZsubmittedfont}{\it}
  \newcommand{\ZZdegreefont}{\bf}
  \newbox\ZZboxa
  \prop_clear:N \l_tmpa_prop
  \prop_put:Nnn \l_tmpa_prop {iu-bl-musice}  {Department~of~Music~Education}
  \prop_put:Nnn \l_tmpa_prop {iu-bl-musico}  {Department~of~Musicology}
  \prop_put:Nnn \l_tmpa_prop {iu-bl-musict}  {Department~of~Music~Theory}
  \prop_put:Nnn \l_tmpa_prop {pu-fw-ap}      {Department~of~Physics}
  \prop_put:Nnn \l_tmpa_prop {pu-fw-bio}     {Department~of~Biological~Sciences}
  \prop_put:Nnn \l_tmpa_prop {pu-fw-cme?}    {Department~of~Civil~and~Mechanical~Engineering}
  \prop_put:Nnn \l_tmpa_prop {pu-fw-comm}    {Department~of~Communication}
  \prop_put:Nnn \l_tmpa_prop {pu-fw-cs}      {Department~of~Computer~Science}
  \prop_put:Nnn \l_tmpa_prop {pu-fw-ece}     {Department~of~Electrical~and~Computer~Engineering}
  \prop_put:Nnn \l_tmpa_prop {pu-fw-engl}    {Department~of~English}
  \prop_put:Nnn \l_tmpa_prop {pu-ha-bio}     {Department~of~Biology}
  \prop_put:Nnn \l_tmpa_prop {pu-ha-cit}     {Department~of~Computer~and~Information~Technology}
  \prop_put:Nnn \l_tmpa_prop {pu-ha-ece}     {Department~of~Electrical~and~Computer~Engineering}
  \prop_put:Nnn \l_tmpa_prop {pu-ha-hdfs}    {Department~of~Human~Development~and~Family~Science}
  \prop_put:Nnn \l_tmpa_prop {pu-ha-hist}    {Department~of~History~and~Philosphy}
  \prop_put:Nnn \l_tmpa_prop {pu-ha-mce}     {Department~of~Mechnical~and~Civil~Engineering}
  \prop_put:Nnn \l_tmpa_prop {pu-in-bio}     {Department~of~Biology}
  \prop_put:Nnn \l_tmpa_prop {pu-in-bme}     {Department~of~Biomedical~Engineering}
  \prop_put:Nnn \l_tmpa_prop {pu-in-ccb?}    {Department~of~Chemistry~and~Chemical~Biology}
  \prop_put:Nnn \l_tmpa_prop {pu-in-cis?}    {Department~of~Computer~and~Information~Science}
  \prop_put:Nnn \l_tmpa_prop {pu-in-ece}     {Department~of~Electrical~and~Computer~Engineering}
  \prop_put:Nnn \l_tmpa_prop {pu-in-fis?}    {Department~of~Forensic~and~Investigative~Sciences}
  \prop_put:Nnn \l_tmpa_prop {pu-in-math}    {Department~of~Mathematical~Sciences}
  \prop_put:Nnn \l_tmpa_prop {pu-in-me}      {Department~of~Mechanical~and~Energy~Engineering}
  \prop_put:Nnn \l_tmpa_prop {pu-in-mste}    {School~of~Engineering~and~Technology}
  \prop_put:Nnn \l_tmpa_prop {pu-in-phys}    {Department~of~Physics}
  \prop_put:Nnn \l_tmpa_prop {pu-we-bio}     {Department~of~Biology}
  \prop_put:Nnn \l_tmpa_prop {pu-we-ece}     {Department~of~Electrical~and~Computer~Engineering}
  \prop_put:Nnn \l_tmpa_prop {pu-we-hdfs}    {Department~of~Human~Development~and~Family~Science}
  \prop_put:Nnn \l_tmpa_prop {pu-we-hist}    {Department~of~History~and~Philosphy}
  \prop_put:Nnn \l_tmpa_prop {pu-we-mce}     {Department~of~Mechnical~and~Civil~Engineering}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-aae}     {School~of~Aeronautics~and~Astronautics}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-abe}     {School~of~Agricultural~and~Biological~Engineering}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-ad}      {Patti~\&~Rusty~Rueff~School~of~Design,~Art~and~Performance}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-agecon}  {Department~of~Agricultural~Economics}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-agry}    {Department~of~Agronomy}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-amst}    {Department~of~American~Studies}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-ansc}    {Department~of~Animal~Sciences}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-anth}    {Department~of~Anthropology}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-asec}    {Department~of~Agricultural~Sciences~Education~and~Communication}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-bchm}    {Department~of~Biochemistry}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-biol}    {Department~of~Biological~Sciences}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-bme}     {Weldon~School~of~Biomedical~Engineering}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-bms}     {Department~of~Basic~Medical~Sciences}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-btny}    {Department~of~Botany~and~Plant~Pathology}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-cce}     {Lyles~School~of~Civil~and~Construction~Engineering}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-cgt}     {Department~of~Computer~Graphics~Technology}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-che}     {Davidson~School~of~Chemical~Engineering}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-chem}    {Department~of~Chemistry}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-cit}     {Department~of~Computer~and~Information~Technology}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-cm}      {School~of~Construction~Management~Technology}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-cmpl}    {School~of~Interdisciplinary~Studies~(Comparitive~Literature)}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-com}     {Brian~Lamb~School~of~Communications}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-cpb}     {Department~of~Comparitive~Pathobiology}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-cs}      {Department~of~Computer~Science}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-csr}     {Department~of~Consumer~Science}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-eaps}    {Department~of~Earth,~Atmospheric,~and~Planetary~Sciences}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-ece}     {Elmore~Family~School~of~Electrical~and~Computer~Engineering}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-econ}    {Department~of~Economics}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-edci}    {Department~of~Curriculum~and~Instruction}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-edst}    {Department~of~Educational~Studies}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-eee}     {Division~of~Environmental~and~Ecological~Engineering}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-ene}     {School~of~Engineering~Education}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-engl}    {Department~of~English}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-entm}    {Department~of~Entomology}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-et}      {School~of~Engineering~Technology}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-fnr}     {Department~of~Forestry~and~Natural~Resources}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-fs}      {Department~of~Food~Science}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-hdfs}    {Department~of~Human~Development~and~Family~Science}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-hist}    {Department~of~History}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-hk}      {School~of~Health~and~Kinesiology}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-hort}    {Department~of~Horticulture}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-hsci}    {School~of~Health~Sciences}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-htm}     {Department~of~Hospitality~and~Tourism~Management}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-ibsc}    {Interdisciplinary~Biomedical~Studies~Program}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-ie}      {Edwardson~School~of~Industrial~Engineering}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-ipph}    {Department~of~Industrial~and~Physical~Pharmacy}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-lc}      {School~of~Languages~and~Cultures}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-ling}    {Department~of~Linguistics}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-math}    {Department~of~Mathematics}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-mcmp}    {Department~of~Medicinal~Chemistry~and~Molecular~Pharmacology}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-me}      {School~of~Mechanical~Engineering}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-mgmt}    {School~of~Management}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-mse}     {School~of~Materials~Engineering}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-ne}      {School~of~Nuclear~Engineering}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-nur}     {School~of~Nursing}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-nutr}    {Department~of~Nutrition~Science}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-obhr}    {Department~of~Organizational~Behavior~and~Human~Resource~Management}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-owl}     {Online~Writing~Lab}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-phil}    {Department~of~Philosophy}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-phillit} {Department~of~Philosophy~and~Literature}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-phpr}    {Department~of~Pharmacy~Practice}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-phys}    {Department~of~Physics~and~Astronomy}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-pol}     {Department~of~Political~Science}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-psd}     {Purdue~Thesis~Software~Development}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-psy}     {Department~of~Psychological~Sciences}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-pubh}    {Department~of~Public~Health}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-satt}    {School~of~Aviation~and~Transportation~Technology}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-slhs}    {Department~of~Speech,~Language,~and~Hearing~Sciences}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-soc}     {Department~of~Sociology}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-stat}    {Department~of~Statistics}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-tdo}     {Thesis~and~Dissertation~Office}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-tech}    {Polytechnic~Institute}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-thtr}    {Department~of~Theatre}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-tli}     {Department~of~Technology~Leadership~and~Innovation}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-vcs}     {Department~of~Veterinary~Clinical~Sciences}
  \prop_put:Nnn \l_tmpa_prop {pu-wl-ydae}    {Department~of~Youth~Development~and~Agricultural~Education}
  \prop_put:Nnn \l_tmpa_prop {uh-ma-econ}    {Department~of~Economics}
  \prop_put:Nnn \l_tmpa_prop {unr-re-me}     {Mechanical~Engineering~Department}
  \prop_get:NVNTF
    \l_tmpa_prop
    \ZZinscampro
    \ZZwholename
    % TRUE CODE
    { }
    % FALSE CODE
    { }

  \renewcommand*{\maketitle}
    {
      \pdfbookmark{TITLE~PAGE}{titlepage}
      \str_case:VnTF
        {\ZZins}
        {
          %
          %  Purdue University cover page.
          %
          {pu}
            {
              % Center vertically on page.
              % \ZZpresentandinorder{title}%
              %   {
              %     \setcounter{ZZcountersavedpagenumber}{\value{page}}%
              %     \thispagestyle{empty}%
              %     \setbox\ZZboxa\vbox
              %       {
              %         \centering
              %           {\ZZtitlefont\uppercase\expandafter{\ZZtitle}}\\
              %           by\\
              %           {\ZZauthorfont\ZZauthor}\\
              %           \vspace*{25pt}
              %           {\ZZdocumentfont \ZZdocument}\\
              %           {\ZZsubmittedfont Submitted~to~the~Faculty~of~Purdue~University}\\
              %           {\ZZsubmittedfont In~Partial~Fulfillment~of~the~Requirements~for~the~degree~of}\\
              %           \vspace*{25pt}
              %           {\ZZdegreefont\ZZdegree}\\
              %           \vspace*{35pt}
              %           % Use the improved vector Purdue seal from Ting-Wei Liu.
              %           \hbox to\textwidth{\hss\includegraphics[width=7.9cm]{gr-pu-seal-vector.pdf}\hss}
              %           \vspace*{25pt}
              %           \ZZwholename\ZZatinformation\\
              %           \ZZcampus,~Indiana\\
              %           \ZZgraduation
              %       }
              %     \setlength{\ZZlengtha}{\textheight}%
              %     \addtolength{\ZZlengtha}{-\ht\ZZboxa}%
              %     \setlength{\ZZlengtha}{0.5\ZZlengtha}%
              %     \vspace*{\ZZlengtha}%
              %     \vspace*{-0.15in}%
              %     \box\ZZboxa
              %   }
              % Use 1.5" top margin.
              \ZZpresentandinorder{title}%
                {
                  \setcounter{ZZcountersavedpagenumber}{\value{page}}%
                  \thispagestyle{empty}%
                  \setbox\ZZboxa\vbox
                    {
                      \centering
                        {\ZZtitlefont\uppercase\expandafter{\ZZtitle}}\\
                        by\\
                        {\ZZauthorfont\ZZauthor}\\
                        \vspace*{25pt}
                        {\ZZdocumentfont \ZZdocument}\\
                        {\ZZsubmittedfont Submitted~to~the~Faculty~of~Purdue~University}\\
                        {\ZZsubmittedfont In~Partial~Fulfillment~of~the~Requirements~for~the~degree~of}\\
                        \vspace*{25pt}
                        {\ZZdegreefont\ZZdegree}\\
                        \vspace*{35pt}
                        % Use the improved vector Purdue seal from Ting-Wei Liu.
                        \hbox to\textwidth{\hss\includegraphics[width=7.9cm]{gr-pu-seal-vector.pdf}\hss}
                        \vspace*{25pt}
                        \ZZwholename\ZZatinformation\\
                        \ZZcampus,~Indiana\\
                        \ZZgraduation
                    }
                  \vspace*{-0.15in}
                  \vspace*{0.5in}
                  \box\ZZboxa
                }
            }
          %
          %  University of Hawaii cover page.
          %
          {uh}
            {
              \ZZpresentandinorder{title}%
                {
                  \setcounter{ZZcountersavedpagenumber}{\value{page}}%
                  \thispagestyle{empty}%
                  \setbox\ZZboxa\vbox
                    {
                      \centering
                        {\bfseries \Large \uppercase\expandafter{\ZZtitle}}\\[-6pt]
                        \hbox to\textwidth{\hss\includegraphics[width=7cm]{gr-uh-seal.pdf}\hss}
                        \vspace*{-6pt}
                        {\large \ZZdocument\ submitted~to~the~Graduate~Division~of~the}\\
                        {\large University~of~Hawai\okina i~at~M\=anoa~in~partial~fulfillment~of~the}\\
                        {\large requirements~for~the~degree~of}\\
                        \vspace*{18pt}
                        {\bfseries \large \ZZdegree}\\
                        {\large in}\\
                        {\bfseries \large \ZZprogram}\\
                        \vspace*{18pt}
                        {\bfseries \large \ZZgraduation}\\
                        \vspace*{18pt}
                        {\large by}\\
                        {\bfseries \large \ZZauthor}\\
                        \vspace*{18pt}
                        {\large Dissertation~Committee:}\\[-6pt]
                        \clist_set_eq:NN \l_tmpa_clist \ZZcommittee
                        \clist_map_inline:Nn \l_tmpa_clist {\large ##1\\[-6pt]}
                        \vspace*{24pt}
                        {\large Keywords:~\ZZkeywords}
                    }
                  \setlength{\ZZlengtha}{\textheight}%
                  \addtolength{\ZZlengtha}{-\ht\ZZboxa}%
                  \setlength{\ZZlengtha}{0.5\ZZlengtha}%
                  \vspace*{\ZZlengtha}%
                  \vspace*{-0.15in}%
                  \box\ZZboxa
                }
            }
        }
        % TRUE CODE
        { }
        % FALSE CODE
        { }
      \setcounter{page}{\value{ZZcountersavedpagenumber}}%
    }

  \newcommand{\ZZnonchapter}[4]
    {
      {
        \ZZnottableofcontents
        \group_begin:
          \bool_set_true:N \ZZinotherb
          \chapter*{#2}
          \ifthen{\equal{y}{#3}}
            {\addcontentsline{toc}{chapter}{#2}}
        \group_end:
        \vspace{#4}
      }
      \bool_if:NTF \ZZinappendixb
        % TRUE CODE
        {\renewcommand{\thesection}{\thechapter.\arabic{section}}}
        % FALSE CODE
        {\renewcommand{\thesection}{\arabic{chapter}.\arabic{section}}}
    }

  \def\@starttoc#1
    {
      \begingroup
        \@input{\jobname.#1}
        \if@filesw \expandafter\newwrite\csname tf@#1\endcsname
          \immediate\openout \csname tf@#1\endcsname \jobname.#1\relax
        \fi
        \tex_global:D\@nobreakfalse
      \endgroup
    }

  \renewcommand*{\tableofcontents}
    {
      \ZZpresentandinorder{tableofcontents}
      \ZZnonchapter{odd}{\ZZtableofcontentsname}{n}{0pt}
      {
        \output=
          {
            % \let \par \ZZpar
            \ifnum \outputpenalty<-\@M
              \@specialoutput
            \else
              \@makecol
              \@opcol
              \@startcolumn
              \@whilesw \if@fcolmade \fi
              {\@opcol\@startcolumn}
            \fi
            \ifnum \outputpenalty>-\@Miv
              \ifdim \@colroom<1.5\baselineskip
                \ifdim \@colroom<\textheight
                  \@latex@warning@no@line{Text page \thepage+contains only floats}
                  \@emptycol
                \else
                  \tex_global:D \vsize \@colroom
                \fi
              \else
                \tex_global:D \vsize \@colroom
              \fi
            \else
              \tex_global:D \vsize \maxdimen
            \fi
            % {\leftskip=0pt \noindent\hbox to\textwidth{\hfil Page}\endgraf}
          }
          \ZZbaselinestretch{1}
          \@starttoc{toc}
      }
    }

  \renewcommand*{\listoftables}
    {
      \ZZpresentandinorder{listoftables}
      \ZZnonchapter{next}{\ZZlistoftablesname}{y}{0pt}
      {
        \output=
          {
            % \let \par \ZZpar
            \ifnum \outputpenalty<-\@M
              \@specialoutput
            \else
              \@makecol
              \@opcol
              \@startcolumn
              \@whilesw \if@fcolmade \fi
              {\@opcol\@startcolumn}
            \fi
            \ifnum \outputpenalty>-\@Miv
              \ifdim \@colroom<1.5\baselineskip
                \ifdim \@colroom<\textheight
                  \@latex@warning@no@line{Text page \thepage+contains only floats}
                  \@emptycol
                \else
                  \tex_global:D \vsize \@colroom
                \fi
              \else
                \tex_global:D \vsize \@colroom
              \fi
            \else
              \tex_global:D \vsize \maxdimen
            \fi
          }
        \ZZbaselinestretch{1}
        \@starttoc{lot}
      }
    }

  \newcommand*{\ZZlistoflistings}
    {
      \ZZpresentandinorder{ZZlistoflistings}
      \ZZnonchapter{next}{\ZZlistoflistingsname}{y}{0pt}
      {
        \output=
          {
            % \let \par \ZZpar
            \ifnum \outputpenalty<-\@M
              \@specialoutput
            \else
              \@makecol
              \@opcol
              \@startcolumn
              \@whilesw \if@fcolmade \fi
              {\@opcol\@startcolumn}
            \fi
            \ifnum \outputpenalty>-\@Miv
              \ifdim \@colroom<1.5\baselineskip
                \ifdim \@colroom<\textheight
                  \@latex@warning@no@line{Text page \thepage+contains only floats}
                  \@emptycol
                \else
                  \tex_global:D \vsize \@colroom
                \fi
              \else
                \tex_global:D \vsize \@colroom
              \fi
            \else
              \tex_global:D \vsize \maxdimen
            \fi
          }
        \ZZbaselinestretch{1}
        \@starttoc{lol}
      }
    }

  \newcommand*{\listofprotocols}
    {
      \ZZpresentandinorder{listofprotocols}
      \ZZnonchapter{next}{\ZZlistofprotocolsname}{y}{0pt}
      {
        \output=
          {
            % \let \par \ZZpar
            \ifnum \outputpenalty<-\@M
              \@specialoutput
            \else
              \@makecol
              \@opcol
              \@startcolumn
              \@whilesw \if@fcolmade \fi
              {\@opcol\@startcolumn}
            \fi
            \ifnum \outputpenalty>-\@Miv
              \ifdim \@colroom<1.5\baselineskip
                \ifdim \@colroom<\textheight
                  \@latex@warning@no@line{Text page \thepage+contains only floats}
                  \@emptycol
                \else
                  \tex_global:D \vsize \@colroom
                \fi
              \else
                \tex_global:D \vsize \@colroom
              \fi
            \else
              \tex_global:D \vsize \maxdimen
            \fi
          }
        \ZZbaselinestretch{1}
        \@starttoc{lop}
      }
    }

  \newcommand*{\listofschemes}
    {
      \ZZpresentandinorder{listofschemes}
      \ZZnonchapter{next}{\ZZlistofschemesname}{y}{0pt}
      {
        \output=
          {
            % \let \par \ZZpar
            \ifnum \outputpenalty<-\@M
              \@specialoutput
            \else
              \@makecol
              \@opcol
              \@startcolumn
              \@whilesw \if@fcolmade \fi
              {\@opcol\@startcolumn}
            \fi
            \ifnum \outputpenalty>-\@Miv
              \ifdim \@colroom<1.5\baselineskip
                \ifdim \@colroom<\textheight
                  \@latex@warning@no@line{Text page \thepage\space contains only floats}
                  \@emptycol
                \else
                  \tex_global:D \vsize \@colroom
                \fi
              \else
                \tex_global:D \vsize \@colroom
              \fi
            \else
              \tex_global:D \vsize \maxdimen
            \fi
          }
        \ZZbaselinestretch{1}
        \@starttoc{los}
      }
    }

  \newcommand{\ZZStartList}[1]
    {
      \ZZnonchapter{odd}{#1}{y}{0pt}
      \setlength{\LTleft}{0truein}
      \setlength{\LTright}{0truein}
      %% ! change to \setlength[\ZZtl}{\textwidth - \LTleft - \LTright - 2*\tabcolsep - 1truein} later
      \setlength{\ZZtl}{\textwidth}
      \addtolength{\ZZtl}{-\LTleft}
      \addtolength{\ZZtl}{-\LTright}
      \addtolength{\ZZtl}{-\tabcolsep}
      \addtolength{\ZZtl}{-\tabcolsep}
      \addtolength{\ZZtl}{-1truein}
      %% ! Get real width of first column.
      %% ! \setcounter{\ZZcount@temp}{\c@LT@tables}
      %% ! \addtocounter(\ZZcount@temp){1}
      %% ! \def\stuff{\csname LT@\romannumeral\ZZcount@temp\endcsname}
      %% ! \showthe\stuff
      %% ! \addtolength{\ZZtl}{-?}
    }

  \newcommand{\chapterappendix}
    {
      \par
      \bool_set_true:N \ZZinchapterappendixb
      \setcounter{section}{0}
      \bool_if:NTF \ZZinappendixb
        {\renewcommand{\thesection}{\thechapter.\AlphAlph{\value{section}}}}
        {\renewcommand{\thesection}{\thechapter.\arabic{section}}}
    }

  \newcommand{\chapterappendices}
    {
      \par
      \bool_set_true:N \ZZinchapterappendixb
      \setcounter{section}{0}
      \bool_if:NTF \ZZinappendixb
        {\renewcommand{\thesection}{\thechapter.\AlphAlph{\value{section}}}}
        {\renewcommand{\thesection}{\thechapter.\arabic{section}}}
    }

  \renewcommand*{\listoffigures}
    {
      \ZZpresentandinorder{listoffigures}
      \ZZnonchapter{next}{\ZZlistoffiguresname}{y}{0pt}
      {
        \output=
          {
            %%%%                \let \par \ZZpar
            \ifnum \outputpenalty<-\@M
              \@specialoutput
            \else
              \@makecol
              \@opcol
              \@startcolumn
              \@whilesw \if@fcolmade \fi
              {\@opcol\@startcolumn}
            \fi
            \ifnum \outputpenalty>-\@Miv
              \ifdim \@colroom<1.5\baselineskip
                \ifdim \@colroom<\textheight
                  \@latex@warning@no@line{Text page \thepage\space contains only floats}
                  \@emptycol
                \else
                  \tex_global:D \vsize \@colroom
                \fi
              \else
                \tex_global:D \vsize \@colroom
              \fi
            \else
              \tex_global:D \vsize \maxdimen
            \fi
          }
        \ZZbaselinestretch{1}
        \@starttoc{lof}
      }
    }

  \newcommand{\ZZPause}
    {
      \typeout{Press▯Enter▯key▯to▯continue:▯}
      \read -1 to \trash
    }

  \newcommand{\ZZRepeat}[2]
    {
      \foreach \ZZATemp in {1,..,#2}
      {
        #1
      }
    }

  \newenvironment{abbreviations}
    {
      \ZZpresentandinorder{abbreviations}
      \ZZStartList{\ZZabbreviationsname}
      \xltabular{\textwidth}{@{}lX@{}}
    }
    {
      \endxltabular
    }

  \renewenvironment{abstract}
    {
      \ZZpresentandinorder{abstract}
      \ZZnonchapter{next}{\ZZabstractname}{y}{0pt}
      %%%%        \vspace*{\baselineskip}
    }
    {
      \endgraf
    }

  \newenvironment{acknowledgments}
    {
      \pdfbookmark{ACKNOWLEDGMENTS}{acknowledments}
      \ZZpresentandinorder{acknowledgments}
      \ZZnonchapter{next}{\ZZacknowledgmentsname}{n}{0pt}
    }
    { }

  \newenvironment{cabstract}
    {
      \begin{quote}
        \textbf{\ZZabstractname}\quad
    }
    {
      \end{quote}
    }

  \newenvironment{cland}
    {
      \begin{landscape}
        \hbox\bgroup
          \hss
          \vbox\bgroup
    }
    {
          \egroup
          \hss
        \egroup
      \end{landscape}
    }

  \newenvironment{dedication}
    {
      \pdfbookmark{DEDICATION}{dedication}
      \ZZpresentandinorder{dedication}
      \newpage
      \mbox{}
      \vfil
      \begin{center}
    }
    {
      \end{center}
      \vfil
      \eject
      \ZZnottableofcontents
    }

  \renewenvironment{glossary}
    {
      \ZZpresentandinorder{glossary}
      \ZZStartList{\ZZglossaryname}
      \xltabular{\textwidth}{@{}lX@{}}
    }
    {
      \endxltabular
    }

  \newenvironment{lquotation}
    {
      \begin{quotation}
        \ZZbaselinestretch{1}
    }
    {
      \end{quotation}
    }

  \newenvironment{nomenclature}
    {
      \ZZpresentandinorder{nomenclature}
      \ZZStartList{\ZZnomenclaturename}
      \xltabular{\textwidth}{@{}lX@{}}
    }
    {
      \endxltabular
    }


  \newenvironment{preface}
    {
      \pdfbookmark{PREFACE}{preface}
      \ZZpresentandinorder{preface}
      \ZZnonchapter{next}{\ZZprefacename}{n}{0pt}
    }
    { }


  \newenvironment{statement}
    {
      \pdfbookmark{COMMITTEE~APPROVAL}{committeeapproval}
      \def\approvedby##1
        {
          \vspace*{35pt}
          \centering
            \textbf{Approved~by:}\\
            ##1
          \endgraf
        }
      \def\entry##1##2
        {
          \vspace*{25pt}
          \centering
            \textbf{##1}\\
            ##2\\
            \endgraf
        }
      \newpage
      \vspace*{0.2truein}
      \centering
        {\large\bf THE~PURDUE~UNIVERSITY~GRADUATE~SCHOOL}\\
        {\large\bf STATEMENT~OF~COMMITTEE~APPROVAL}\\
        \vspace*{10pt}
    }
    { }

  \newenvironment{symbols}
    {
      \ZZpresentandinorder{symbols}
      \ZZStartList{\ZZsymbolsname}
      \xltabular{\textwidth}{@{}lX@{}}
    }
    {
      \endxltabular
    }

  \renewcommand{\bottomfraction}{1}
  \renewcommand{\floatpagefraction}{0}
  \renewcommand{\topfraction}{1}
  \setcounter{bottomnumber}{10}
  \setcounter{topnumber}{10}
  \setcounter{totalnumber}{10}
  \newboolean{ZZFigureCaptionsB}
  \setboolean{ZZFigureCaptionsB}{false}

  \renewcommand{\chapter}
    {
      \bool_if:nTF
        {
          (\ZZinotherb  &&  \ZZNonChapterBlankPagesB)
          ||
          (!\ZZinotherb  &&  \ZZChapterBlankPagesB)
        }
        % TRUE CODE
        {\cleardoublepage}
        % FALSE CODE
        {\clearpage}
      \ZZnottableofcontents
      \tex_global:D \@topnum=\z@
      \@afterindentfalse
      \bool_if:NTF \ZZinappendixb
        % TRUE CODE
        {\renewcommand{\thesection}{\thechapter.\arabic{section}}}
        % FALSE CODE
        {\renewcommand{\thesection}{\thechapter.\arabic{section}}}
      \secdef \@chapter \@schapter
    }

  \newcommand{\coversheet}[1]
    {
      {
        \bool_if:NT \ZZCoverSheetsB
          {
            \bool_if:NTF \ZZChapterBlankPagesB
              % TRUE CODE
              {\clearpage}
              % FALSE CODE
              {\clearpage}
            \pagestyle{empty}
            \mbox{}
            \ZZCoversheetSpace
            \begin{center}
              #1
            \end{center}
            \vfill
            \newpage
            \addtocounter{page}{-1}
          }
        \ZZnottableofcontents
      }
    }

  \renewcommand{\appendix}
    {
      \ZZpresentandinorder{appendix}
      \par
      \bool_set_true:N \ZZinappendixb
      \setcounter{chapter}{0}
      \setcounter{section}{0}
      \coversheet{\uppercase\expandafter{\ZZAppendixName}}
      \renewcommand{\@chapapp}{\ZZAppendixName}
      \renewcommand{\thechapter}{\AlphAlph{\value{chapter}}}
    }

  \newcommand{\ZZAppendicesName}{Appendices}
  \newcommand{\appendices}
    {
      \ZZpresentandinorder{appendices}
      \par
      \bool_set_true:N \ZZinappendixb
      \setcounter{chapter}{0}
      \setcounter{section}{0}
      \coversheet{\uppercase\expandafter{\ZZAppendicesName}}
      \renewcommand{\@chapapp}{\ZZAppendicesName}
      \renewcommand{\thechapter}{\AlphAlph{\value{chapter}}}
    }

  \newenvironment{notesandfootnotes}
    {\ZZpresentandinorder{notesandfootnotes}}
    {\par}

  \newenvironment{vita}
    {
      \ZZpresentandinorder{vita}
      \ZZVitaStart
      \ZZnonchapter{odd}{\ZZVitaName}{y}{0pt}
    }
    { }

  \newenvironment{publication}
    {
      \ZZpresentandinorder{publication}
      \ZZPublicationStart
      \ZZnonchapter{odd}{\ZZPublicationName}{y}{0pt}
    }
    { }

  \newenvironment{publications}
    {
      \ZZpresentandinorder{publications}
      \ZZPublicationStart
      \ZZnonchapter{odd}{\ZZPublicationsName}{y}{0pt}
    }
    { }

  \newcounter{ZZi}
  \newcounter{ZZnameindex}
  \newlength{\ZZcaptionwidth}
  \newlength{\ZZpaddinglength}
  \newlength{\ZZparindent}
  \setlength{\ZZparindent}{\parindent}
  \newlength{\ZZtl}
  \newlength{\ZZtlb}
  \setlength{\evensidemargin}{0.5truein}
  \settoheight{\headheight}{- ivx1 -}
  \setlength{\headsep}{0.5truein}
  \addtolength{\headsep}{-\headheight}
  \addtolength{\headsep}{-0.0625truein}

  \def\@makechapterhead#1
    {
      % !!! \ZZchapterheadspacea is now 0pt.
      % !!! \vspace*{\ZZchapterheadspacea}
      {
        \centering
          \ZZmake@chapter@head{#1}
          \endgraf
      }
      \vspace*{\ZZchapterheadspaceb}
    }

  \long\def\@makefntext#1
    {
      \baselineskip=12pt
      \noindent
      \@makefnmark #1
    }

  \def\@makeschapterhead#1
    {
      % !!! \ZZschapterheadspacea is now 0pt.
      % !!! \vspace*{\ZZschapterheadspacea}
      {
        \centering
          \ZZmake@schapter@head{#1}
          \endgraf
      }
      \vspace*{\ZZschapterheadspaceb}
    }

  \def\@sect#1#2#3#4#5#6[#7]#8
    {
      \ifnum #2>\c@secnumdepth
        \let\@svsec\@empty
      \else
        \refstepcounter{#1}
        \edef\@svsec{\csname the#1\endcsname\hskip 1em}
      \fi
      \ifthen{\equal{che}{\ZZprogram}}
        {
          \ifnum #2=3
            \refstepcounter{#1}
            \edef\@svsec{\csname the#1\endcsname\hskip 1em}
          \fi
        }
      \@tempskipa #5\relax
      \ifdim \@tempskipa>\z@
        \begingroup
          #6\relax
          \@hangfrom{\hskip #3\relax\@svsec}
          {\interlinepenalty \@M #8\par}
        \endgroup
        \csname #1mark\endcsname{#7}
        \addcontentsline{toc}{#1}
          {
            \ifnum #2>\c@secnumdepth
            \else
              \protect\numberline{\csname the#1\endcsname}
            \fi
            #7
          }
      \else
        \def\@svsechd
          {
            #6
            \hskip #3\relax
            \@svsec #8\csname #1mark\endcsname
            {#7}
            \addcontentsline{toc}{#1}
            {
              \ifnum #2>\c@secnumdepth
              \else
                \protect\numberline{\csname the#1\endcsname}
              \fi
              #7
            }
          }
      \fi
      \@xsect{#5}
    }

  \def\@startsection#1#2#3#4#5#6
    {
      \if@noskipsec
        \leavevmode
      \fi
      \par
      \@tempskipa #4\relax
      \@afterindenttrue
      \ifdim \@tempskipa <\z@
        \@tempskipa -\@tempskipa
      \fi
      \if@nobreak
        \everypar{}
      \else
        \addpenalty{\@secpenalty}
        \addvspace{\@tempskipa}
      \fi
      \@ifstar
        {\@ssect{#3}{#4}{#5}{#6}}
        {\@dblarg{\@sect{#1}{#2}{#3}{#4}{#5}{#6}}}
    }

  \renewcommand{\section}{\@startsection{section}{1}{\z@}{24pt}{12pt}{\ZZbaselinestretch{1}\ZZsectionseries}}
  \renewcommand{\subsection}{\@startsection{subsection}{2}{\z@}{24pt}{12pt}{\ZZbaselinestretch{1}\ZZsubsectionseries}}
  \renewcommand{\subsubsection}{\@startsection{subsubsection}{3}{\z@}{24pt}{12pt}{\ZZbaselinestretch{1}\ZZsubsubsectionseries}}
  \renewcommand{\paragraph}{\@startsection{paragraph}{4}{\z@}{24pt}{12pt}{\ZZbaselinestretch{1}\ZZparagraphseries}}
  \renewcommand{\subparagraph}{\@startsection{subparagraph}{5}{\parindent}{24pt}{12pt}{\ZZbaselinestretch{1}\ZZsubparagraphseries}}
  \renewcommand{\baselinestretch}{1.5}
  \pagenumbering{arabic}
  \raggedbottom

  \NeedsTeXFormat{LaTeX2e}

  \setcounter{chapter}{0}
  \newcommand{\ZZAppendixName}{Appendix}

  \newcommand{\articlepages}[1]
    {
      \ZZRepeat{\ZZBlankPage}{#1}
    }

  \newcommand{\articleinclude}[1]
    {
      \newcommand{\ZZATemp}{#1}
      \foreach \ZZBTemp in {1,..,1000}
        {
          \newcommand{\ZZCTemp}{\ZZATemp\ZZBTemp.eps}
          \typeout{\ZZCTemp}
          \IfFileExists{\ZZCTemp}
            {\noindent\includegraphics[width=\textwidth]{\ZZCTemp}\newpage}
            {\breakforeach}
        }
    }

  \def\@chapter[#1]#2
    {
      \bool_if:NTF \ZZinappendixb
        % TRUE CODE
        {\ZZpresentandinorder{achapter}}
        % FALSE CODE
        {\ZZpresentandinorder{chapter}}
      \ifnum \c@secnumdepth >\m@ne
        \refstepcounter{chapter}
        \typeout{\@chapapp\space\thechapter.}
        \bool_set_true:N \ZZUnsetB
        \ifthen{\equal{agec}{\ZZprogram}}
          {\bool_set_false:N \ZZUnsetB\addcontentsline{toc}{chapter}{\protect\uppercase{\@chapapp\ \thechapter. #1}}}
        \ifthen{\equal{cce}{\ZZprogram}}
          {\bool_set_false:N \ZZUnsetB\addcontentsline{toc}{chapter}{\protect\numberline{\thechapter}\uppercase{#1}}}
        \ifthen{\equal{mse}{\ZZprogram}}
          {\bool_set_false:N \ZZUnsetB\addcontentsline{toc}{chapter}{\protect\uppercase{\@chapapp\ \thechapter. #1}}}
        \bool_if:NT \ZZUnsetB
          {\addcontentsline{toc}{chapter}{\protect\numberline{\thechapter}{#1}}}
      \else
        \addcontentsline{toc}{chapter}{#1}
      \fi
      \chaptermark{#1}
      % \addtocontents{lof}{\protect\addvspace{10\p@}}
      % \addtocontents{lot}{\protect\addvspace{10\p@}}
      \@makechapterhead{#2}
      \@afterheading
      \bool_if:NTF \ZZInChaptersB
        % TRUE CODE
        { }
        % FALSE CODE
        {
          %% xx \pagenumbering{arabic}
          \bool_set_true:N \ZZInChaptersB
        }
    }

  \newcommand{\nonchapterblankpages}{\bool_set_true:N \ZZNonChapterBlankPagesB}
  \newcommand{\nononchapterblankpages}{\bool_set_false:N \ZZNonChapterBlankPagesB}
  \newcommand{\chapterblankpages}{\bool_set_true:N \ZZChapterBlankPagesB}
  \newcommand{\nochapterblankpages}{\bool_set_false:N \ZZChapterBlankPagesB}
  \newcommand{\coversheets}{\bool_set_true:N \ZZCoverSheetsB}
  \newcommand{\nocoversheets}{\bool_set_false:N \ZZCoverSheetsB}

  \def\LT@makecaption#1#2#3
    {
      \LT@mcol\LT@cols c{\hbox to\z@{\hss\parbox[t]\LTcapwidth{
        \sbox\@tempboxa{#1{#2 }#3}
        \ifdim\wd\@tempboxa>\hsize
          #1{#2 }#3
        \else
          \hbox to\hsize{\hfil\box\@tempboxa\hfil}
        \fi
        \endgraf\vskip\baselineskip
      }\hss}}
    }

  \newcommand{\ZZoptionendnotes}
    {
      \let\footnote=\endnote
      \renewcommand{\pendnotes}
      {
        \newpage
        \begingroup
          % \Baselinestretch{0.9}
          \setlength{\parindent}{0pt}
          \setlength{\parskip}{1.5ex}
          \renewcommand{\enotesize}{\normalsize}
          \theendnotes
        \endgroup
      }
    }

  \renewcommand*{\l@chapter}{\@dottedtocline{0}{\big@toc@skip}{0em}{1.4em}}
  \renewcommand*{\l@section}{\@dottedtocline{1}{\small@toc@skip}{1.5em}{2.3em}}
  \renewcommand*\l@subsection{\@dottedtocline{2}{\small@toc@skip}{3.8em}{3.2em}}
  \renewcommand*{\l@subsubsection}{\@dottedtocline{3}{\small@toc@skip}{7em}{4.1em}}
  \renewcommand*{\l@paragraph}{\@dottedtocline{4}{\small@toc@skip}{10em}{5em}}
  \renewcommand*{\l@subparagraph}{\@dottedtocline{5}{\small@toc@skip}{12em}{6em}}
  \renewcommand*{\l@figure}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
  \newcommand*{\l@ZZlisting}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
  \renewcommand*{\l@protocol}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
  \renewcommand*{\l@scheme}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
  \renewcommand*{\l@table}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
  \bool_set_true:N \ZZFigureCaptionsB
  \newcommand{\ZZCoversheetSpace}{\vfill}
  \newcommand{\AppendixFigure}{\relax}
  \newcommand{\AppendixTable}{\relax}
  \newcommand{\ZZVitaStart}{\coversheet{\ZZVitaName}}
  \newcommand{\ZZVitaName}{VITA}
  \newcommand{\ZZPublicationStart}{\coversheet{\ZZPublicationName}}
  \newcommand{\ZZPublicationName}{PUBLICATION}
  \newcommand{\ZZPublicationsStart}{\coversheet{\ZZPublicationsName}}
  \newcommand{\ZZPublicationsName}{PUBLICATIONS}
  \setlength{\dblfloatsep}{18pt plus 3pt minus 3pt}
  \setlength{\dbltextfloatsep}{30pt plus 3pt minus 6pt}
  \setlength{\floatsep}{18pt plus 3pt minus 3pt}
  \setlength{\intextsep}{18pt plus 3pt minus 3pt}
  \setlength{\textfloatsep}{30pt plus 3pt minus 6pt}
  \newcommand{\ZZlistingname}{Listing~}
  \newcommand{\ZZprotocolname}{Protocol~}
  \newcommand{\ZZschemename}{Scheme~}
  \renewcommand{\tablename}{Table~}
  \def\fnum@ZZlisting{{\ZZlistingseries\ZZlistingname\space\theZZlisting.}~}
  \def\fnum@protocol{{\ZZprotocolseries\ZZprotocolname\space\theprotocol.}~}
  \def\fnum@scheme{{\ZZschemeseries\ZZschemename\space\thescheme.}~}
  \def\fnum@table{{\ZZtableseries\tablename~\thetable.}~}
  \def\ZZmake@chapter@head#1{\ZZchapterseries \ZZchaptersize \thechapter.~#1}
  \def\ZZmake@schapter@head#1{\ZZchapterseries \ZZchaptersize #1}
  \newcommand{\ZZtitleseries}{\bf}
  \newcommand{\ZZauthorseries}{\bf}
  \newcommand{\ZZdocumentseries}{\bf}
  \newcommand{\ZZsubmittedseries}{\it}
  \newcommand{\ZZdegreeseries}{\bf}
  \newcommand{\ZZtitlesize}{\large}
  \newcommand{\ZZauthorsize}{\normalsize}
  \newcommand{\ZZdocumentsize}{\normalsize}
  \newcommand{\ZZsubmittedsize}{\normalsize}
  \newcommand{\ZZdegreesize}{\normalsize}
  \newcommand{\ZZchapterseries}{\bfseries}
  \newcommand{\ZZchaptersize}{\large}
  \newcommand{\ZZfigureseries}{\bfseries}
  \newcommand{\ZZfiguresize}{\large}
  \newcommand{\ZZlistingseries}{\bfseries}
  \newcommand{\ZZlistingsize}{\large}
  \newcommand{\ZZparagraphseries}{\bfseries}
  \newcommand{\ZZparagraphsize}{\large}
  \newcommand{\ZZprotocolseries}{\bfseries}
  \newcommand{\ZZprotocolsize}{\large}
  \newcommand{\ZZschemeseries}{\bfseries}
  \newcommand{\ZZschemesize}{\large}
  \newcommand{\ZZsectionseries}{\bfseries}
  \newcommand{\ZZsectionsize}{\large}
  \newcommand{\ZZsubsectionseries}{\bfseries}
  \newcommand{\ZZsubsectionsize}{\large}
  \newcommand{\ZZsubsubsectionseries}{\bfseries}
  \newcommand{\ZZsubsubsectionsize}{\large}
  \newcommand{\ZZsubsubsubsectionseries}{\bfseries}
  \newcommand{\ZZsubsubsubsectionsize}{\large}
  \newcommand{\ZZsubparagraphseries}{\bfseries}
  \newcommand{\ZZsubparagraphsize}{\large}
  \newcommand{\ZZtableseries}{\bfseries}
  \newcommand{\ZZtablesize}{\large}
  \newcommand{\ZZFigurePunct}{. }
  \newcommand{\ZZTablePunct}{. }
  \setcounter{tocdepth}{4}
  \def\undertext#1{$\underline{\smash{\hbox{#1}}}$}
  \renewcommand{\figurename}{Figure~}
  \renewcommand{\fnum@figure}{{\ZZfigureseries\figurename~\thefigure.}~}
  \newcommand{\ZZchapterheadspacea}{0.6truein}
  \newcommand{\ZZchapterheadspaceb}{0.5\baselineskip}
  \newcommand{\ZZschapterheadspacea}{0.6truein}
  \newcommand{\ZZschapterheadspaceb}{0.5\baselineskip}
  \str_case:Vn
    {\ZZinscampro}
    {
      {pu-in-bme}
      {
        \renewcommand*{\l@section}{\@dottedtocline{1}{\big@toc@skip}{1.5em}{2.3em}}
        \renewcommand*\l@subsection{\@dottedtocline{2}{\big@toc@skip}{3.8em}{3.2em}}
        \renewcommand*{\l@subsubsection}{\@dottedtocline{3}{\big@toc@skip}{7em}{4.1em}}
        \renewcommand*{\l@paragraph}{\@dottedtocline{4}{\big@toc@skip}{10em}{5em}}
        \renewcommand*{\l@figure}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \renewcommand*{\l@ZZlisting}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \renewcommand*{\l@protocol}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \renewcommand*{\l@scheme}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \renewcommand*{\l@table}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \def\numberline#1{\hb@xt@\@tempdima{#1\hfil}}
      }
      {pu-in-ece}
      {
        \renewcommand*{\l@section}{\@dottedtocline{1}{\big@toc@skip}{1.5em}{2.3em}}
        \renewcommand*\l@subsection{\@dottedtocline{2}{\big@toc@skip}{3.8em}{3.2em}}
        \renewcommand*{\l@subsubsection}{\@dottedtocline{3}{\big@toc@skip}{7em}{4.1em}}
        \renewcommand*{\l@paragraph}{\@dottedtocline{4}{\big@toc@skip}{10em}{5em}}
        \renewcommand*{\l@figure}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \renewcommand*{\l@ZZlisting}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \renewcommand*{\l@protocol}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \renewcommand*{\l@scheme}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \renewcommand*{\l@table}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \def\numberline#1{\hb@xt@\@tempdima{#1\hfil}}
      }
      {pu-wl-aae}
      {
        \bool_set_true:N \ZZFigureCaptionsB
        \def\fnum@protocol{{\ZZprotocolseries\protocolname\space\theprotocol}.~}
        \def\fnum@scheme{{\ZZschemeseries\schemename\space\thescheme}.~}
        \def\fnum@table{{\ZZtableseries\tablename~\thetable}.~}
      }
      {pu-wl-ce}
      {
        \bool_set_true:N \ZZFigureCaptionsB
        \def\fnum@protocol{{\ZZprotocolseries\protocolname\space\theprotocol}.~}
        \def\fnum@scheme{{\ZZschemeseries\schemename\space\thescheme}.~}
        \def\fnum@table{{\ZZtableseries\tablename~\thetable}.~}
      }
      {pu-wl-che}
      {
        \renewcommand*{\l@figure}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \renewcommand*{\l@ZZlisting}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \renewcommand*{\l@protocol}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \renewcommand*{\l@scheme}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \renewcommand*{\l@table}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \def\numberline#1{\hb@xt@\@tempdima{#1\hfil}}
      }
      {pu-wl-chem}
      {
        \setlength{\dblfloatsep}{36pt}
        \setlength{\dbltextfloatsep}{36pt}
        \setlength{\floatsep}{36pt}
        \setlength{\intextsep}{36pt}
        \setlength{\textfloatsep}{36pt}
      }
      {pu-wl-ece}
      {
        \renewcommand*{\l@section}{\@dottedtocline{1}{\small@toc@skip}{1.5em}{2.3em}}
        \renewcommand*{\l@subsection}{\@dottedtocline{2}{\small@toc@skip}{3.8em}{3.2em}}
        \renewcommand*{\l@subsubsection}{\@dottedtocline{3}{\small@toc@skip}{7em}{4.1em}}
        \renewcommand*{\l@paragraph}{\@dottedtocline{4}{\small@toc@skip}{10em}{5em}}
        \renewcommand*{\l@subparagraph}{\@dottedtocline{5}{\small@toc@skip}{12em}{6em}}
        \renewcommand*{\l@figure}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \renewcommand*{\l@ZZlisting}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \renewcommand*{\l@protocol}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \renewcommand*{\l@scheme}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \renewcommand*{\l@table}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \def\numberline#1{\hb@xt@\@tempdima{#1\hfil}}
      }
      {pu-wl-hsci}
      {
        \renewcommand*{\l@figure}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \renewcommand*{\l@ZZlisting}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \renewcommand*{\l@protocol}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \renewcommand*{\l@scheme}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \renewcommand*{\l@table}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \def\numberline#1{\hb@xt@\@tempdima{#1\hfil}}
      }
      {pu-wl-me}
      {
        \renewcommand*{\l@figure}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \renewcommand*{\l@ZZlisting}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \renewcommand*{\l@protocol}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \renewcommand*{\l@scheme}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \renewcommand*{\l@table}{\@dottedtocline{1}{\big@toc@skip}{0em}{2.3em}}
        \bool_set_true:N \ZZFigureCaptionsB
        \def\numberline#1{\hb@xt@\@tempdima{#1\hfil}}
        \newcommand{\fnum@protocol}{{\ZZprotocolseries\protocolname\space\theprotocol}.~}
        \newcommand{\fnum@scheme}{{\ZZschemeseries\schemename\space\thescheme}.~}
        \renewcommand{\fnum@figure}{{\ZZfigureseries\figurename~\thefigure.}~}
      }
      {pu-wl-mse}
      {
        \def\ZZCoversheetSpace{0.53125truein}
        \def\AppendixFigure{\addtocontents{lof}{Appendix Figure\hfil}}
        \def\AppendixTable{\addtocontents{lot}{Appendix Table\hfil}}
        \def\AppendixFigure{\addtocontents{lof}{Appendix Figure\hfil}}
        \def\AppendixTable{\addtocontents{lot}{Appendix Table\hfil}}
        \def\undertext#1{\(\underline{\hbox{#1}}\)}
        \setlength{\parindent}{0.40625truein}
      }
    }

  \long\def\@makecaption#1#2
    {
      \vspace*{\abovecaptionskip}
      \if_true:  %%%% bool_if:N \ZZCenterCaptionB    changed 2021-07-27 by mark
        % center caption
        \setlength{\ZZcaptionwidth}{\textwidth}
        \addtolength{\ZZcaptionwidth}{-4\ZZparindent}

        \ifthen{\equal{figure}{\@captype} \or \equal{table}{\@captype}}  %%%%  \or \boolean{ZZFigureCaptionsB}}
          {
            \sbox\@tempboxa{#1 #2}
            \ZZbaselinestretch{1}
            \ifdim \wd\@tempboxa >\hsize
              \centerline{\parbox[t]{\ZZcaptionwidth}{#1 #2}}
            \else
              \centerline{#1 #2}
            \fi
          }

        \ifthen{\equal{ZZlisting}{\@captype}}
          {
            \sbox\@tempboxa{#1 #2}
            \ZZbaselinestretch{1}
            \ifdim \wd\@tempboxa >\hsize
              \centerline{\parbox[t]{\ZZcaptionwidth}{#1 #2}}
            \else
              \centerline{#1 #2}
            \fi
          }

        \ifthen{\equal{protocol}{\@captype}}
          {
            \sbox\@tempboxa{#1 #2}
            \ZZbaselinestretch{1}
            \ifdim \wd\@tempboxa >\hsize
              \centerline{\parbox[t]{\ZZcaptionwidth}{#1 #2}}
            \else
              \centerline{#1 #2}
            \fi
          }

        \ifthen{\equal{scheme}{\@captype}}
          {
            \sbox\@tempboxa{#1 #2}
            \ZZbaselinestretch{1}
            \ifdim \wd\@tempboxa >\hsize
              \centerline{\parbox[t]{\ZZcaptionwidth}{#1 #2}}
            \else
              \centerline{#1 #2}
            \fi
          }

  %                \ifthen{\equal{table}{\@captype}} %%%%  \or \boolean{ZZTableCaptionsB}}
  %                {
  %                    \ZZbaselinestretch{1}
  %                    \centerline{#1}
  %                    \sbox\@tempboxa{#2}
  %                    \ifdim \wd\@tempboxa >\hsize
  %                        \centerline{\parbox[t]{\ZZcaptionwidth}{#2}}
  %                    \else
  %                        \centerline{#2}
  %                    \fi
  %                }
        \vspace*{\belowcaptionskip}
      \else:
        \setlength{\topsep}{0pt}
        \setlength{\parskip}{0pt}
        \setlength{\partopsep}{0pt}
        \begin{quote}
          \renewcommand{\\}{}
          \ZZbaselinestretch{1}
          \ifthenelse{\equal{aae}{\ZZprogram}}
            {#1. #2\par}
            {
              \ifthenelse{\equal{table}{\@captype}}
                {\hfil\strut #1\hfil\break #2\par}
                {#1 #2\par}
            }
        \end{quote}
      \fi:
      \vskip\belowcaptionskip
    }

  {
    \def\ZZATemp{\string##}
    \ifthen{\equal{\ZZprogram}{\relax}}
      {
        \typeout{}
        \typeout{}
        \typeout{No▯program▯option▯was▯specified▯in▯your▯\string\documentclass▯command▯options.}
        \typeout{See}
        \typeout{▯▯▯▯http://engineering.purdue.edu/\string~mark/puthesis/\ZZATemp document}
        \typeout{for▯the▯valid▯choices.}
        \ZZAbort
      }
    \ifthen{\equal{\ZZdocument}{\relax}}
      {
        \typeout{}
        \typeout{}
        \typeout{No▯document▯option▯was▯specified▯in▯your▯\string\documentclass\▯command▯options.}
        \typeout{See}
        \typeout{▯▯▯▯http://engineering.purdue.edu/\string~mark/puthesis/\ZZATemp document}
        \typeout{for▯the▯valid▯choices.}
        \ZZAbort
      }
  }

  \ifthen{\equal{\ZZtemplate}{1}}
    {
      \def\ZZChapterName{Chapter}
      \def\ZZAppendixName{Appendix}
    }

  \ifthen{\equal{\ZZtemplate}{2}}
    {\def\ZZAppendixName{Appendix}}

  \ifthen{\equal{\ZZtemplate}{3}}
    { }

  \typeout{define▯fig-name,▯fig-punct,▯fig-hspace▯and▯same▯for▯table?}


  % The ZZlisting environment.

  \newcounter{ZZlisting}[chapter]
  \renewcommand{\theZZlisting}
    {
      \ifnum \c@chapter>\z@
        \thechapter.
      \fi
      \@arabic\c@ZZlisting
    }
  \def\fps@ZZlisting{tbp}
  \def\ftype@ZZlisting{4}
  \def\ext@ZZlisting{lol}

  \newenvironment{ZZlisting}
    {\@float{ZZlisting}}
    {\end@float}

  \newenvironment{ZZlisting*}
    {\@dblfloat{ZZlisting}}
    {\end@dblfloat}

  % \newcommand{\ZZlistingname}{Listing}


  % The protocol environment.

  \newcounter{protocol}[chapter]
  \renewcommand{\theprotocol}
    {
      \ifnum \c@chapter>\z@
        \thechapter.
      \fi
      \@arabic\c@protocol
    }
  \def\fps@protocol{tbp}
  \def\ftype@protocol{3}
  \def\ext@protocol{lop}

  \newenvironment{protocol}
    {\@float{protocol}}
    {\end@float}

  \newenvironment{protocol*}
    {\@dblfloat{protocol}}
    {\end@dblfloat}

  \newcommand\protocolname{Protocol}


  % The scheme environment.

  \newcounter{scheme}[chapter]
  \renewcommand{\thescheme}
    {
      \ifnum \c@chapter>\z@
        \thechapter.
      \fi
      \@arabic\c@scheme
    }
  \newcommand{\fps@scheme}{tbp}
  \newcommand{\ftype@scheme}{2}
  \newcommand{\ext@scheme}{los}

  \newenvironment{scheme}
    {\@float{scheme}}
    {\end@float}

  \newenvironment{scheme*}
    {\@dblfloat{scheme}}
    {\end@dblfloat}

  \newcommand\schemename{Scheme}

\ExplSyntaxOff

%!!!! Change when the new biber and biblatex are available.
% Use 'c' instead of 'ca. ' for approximate (circa) dates.
% See
%     https://tex.stackexchange.com/questions/224617/what-is-the-correct-way-to-handle-approximate-dates-in-biblatex
% \renewcommand{\datecircadelim}{}
% \DefineBibliographyStrings{english}{circa = c}
