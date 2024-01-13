\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{myarticle}[2023/12/23 @ Yoimiya]

\newif\if@draft
\@draftfalse

\DeclareOption{draft}{
	\@drafttrue
}

\ProcessOptions\relax

\if@draft
 \LoadClass[draft, 12pt]{article}
\else
 \LoadClass[12pt]{article}
\fi

% 标题信息
\makeatletter
\newcommand{\id}[1]{\def\@id{#1}}
\newcommand{\college}[1]{\def\@college{#1}}
\newcommand{\major}[1]{\def\@major{#1}}

\newcommand{\thetitle}{\@title}
\newcommand{\theTitle}{\@title}
\newcommand{\theauthor}{\@author}
\newcommand{\theid}{\@id}
\newcommand{\thecollege}{\@college}
\newcommand{\themajor}{\@major}
\newcommand{\thedate}{\@date}
\makeatother

% 加载中文
\RequirePackage[no-math]{fontspec}
\RequirePackage[UTF8, zihao=-4]{ctex}
% 设置默认字体
\setCJKmainfont[AutoFakeBold=1.5]{SimSun}
\setmainfont{Times New Roman}
\setsansfont{Arial}
\setCJKfamilyfont{zhkai}[AutoFakeBold=1.5]{simkai.ttf}
\setCJKfamilyfont{zhhei}[AutoFakeBold=1.5]{simhei.ttf}
\newcommand*{\kaiti}{\CJKfamily{zhkai}}

%数学类宏包
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{mathrsfs}
\RequirePackage{amsthm}
\RequirePackage{yhmath} % 大型分界符
\RequirePackage{extarrows} % 长度自适应箭头
\RequirePackage{esint} % 闭合曲线积分符号

% 表格宏包
\RequirePackage{makecell} % 表格内换行
\RequirePackage{arydshln} % 表格画虚线
\RequirePackage{booktabs}  %三线表
%% 三线表设置
\usepackage{caption}
\captionsetup{font={small},labelsep=quad}%文字5号，之间空一个汉字符位。
\captionsetup[table]{font={bf}} %表格表号与表题加粗
\RequirePackage{float}

% 字体宏包
%\RequirePackage{enumerate}  % 带编号列表
\RequirePackage{pifont} % 带圈数字
\RequirePackage{bm} % 同时黑体斜体

% 画图宏包
\RequirePackage{color}
\RequirePackage{graphicx}
\RequirePackage{subfigure}
\RequirePackage{wrapfig}
\RequirePackage{tikz}
\captionsetup[figure]{font={bf}} %图号与标题加粗

% 伪代码
\RequirePackage[ruled, linesnumbered]{algorithm2e}

% 封面页
\RequirePackage{titlesec}
\RequirePackage{titling}

\newcommand{\thetitlepage}{
	\begin{titlepage}
		\centering
		\includegraphics[width=50mm]{./supports/sjtu.png}
		
		\vspace*{3cm}
		
		\begin{spacing}{2}
			\textbf{\heiti \zihao{-1} \thetitle}
		\end{spacing}
		
		\vspace*{3cm}
		
		\begin{spacing}{2.0}
			\zihao{3} \heiti 
			\begin{tabular}{ll}
				\textbf{姓 \qquad 名：} & \theauthor \\
				\textbf{学 \qquad 号：} & \theid \\
				\textbf{学 \qquad 院：} & \thecollege \\
				\textbf{专 \qquad 业：} & \themajor
			\end{tabular}
			
			\vspace*{3cm}
			
			\thedate
		\end{spacing}
	\end{titlepage}
}

\renewcommand{\maketitle}{
	\thispagestyle{plain}
	\begin{center}
		\vspace*{20pt} 
		\begin{spacing}{2}
			\textbf{\heiti \zihao{-1} \thetitle}
			
			{\zihao{3} \heiti
				\theauthor \quad \theid \quad \thecollege
			}
		\end{spacing}
	\end{center}
}

% 页面设置
\RequirePackage{geometry}
\RequirePackage{indentfirst}

\geometry{
	a4paper,
	top=38mm,
	bottom=32mm,
	left=30mm,
	right=30mm
}

% 页眉页脚
\RequirePackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\theTitle}
\fancyhead[R]{\leftmark}
\fancyfoot[C]{\thepage}
\makeatletter %双线页眉
\def\headrule{{\if@fancyplain\let\headrulewidth\plainheadrulewidth\fi%
		\hrule\@height 1.5pt \@width\headwidth\vskip1.5pt%上面线为1pt粗
		\hrule\@height 0.5pt\@width\headwidth  %下面0.5pt粗
		\vskip-2\headrulewidth\vskip-1pt}      %两条线的距离1pt
	\vspace{6mm}}     %双线与下面正文之间的垂直间距
\makeatother

%% 行距
\RequirePackage{setspace}
\renewcommand{\baselinestretch}{1.2}

% 摘要
\RequirePackage{ragged2e}
\renewcommand{\abstractname}{\heiti \zihao{-3} \textbf{摘 \quad 要}}
\renewenvironment{abstract}{
	\center{\abstractname}\par
	~\\
	\RaggedRight\setlength{\parindent}{2em}\par
	}{\par
	~\\
	}
% 关键词
\newcommand{\keywords}{
	~\\
	\noindent \textbf{\heiti 关键词:}
	}
	
%\titleformat{\section}
%	{\heiti \zihao{-2} \bfseries \centering} % 字体样式
%	{第 \thesection 章} % 显示章节编号
%	{1em} % 标题和编号之间的间距
%	{} % 之前的代码
%\titlespacing{\section}{0pt}{30pt}{20pt}
%\titlespacing{\subsection}{0pt}{25pt}{12pt}
%\titlespacing{\subsubsection}{0pt}{12pt}{6pt}

\newcommand{\pref}[1]{(\ref{#1})}

% 编号前加入节序号
\numberwithin{equation}{section} 
\numberwithin{figure}{section}
\numberwithin{table}{section}
\allowdisplaybreaks[4]  % 多行公式中换页

% 定理环境
\RequirePackage{thmtools}
\renewcommand{\proofname}{\indent\textbf{Pf}}
\declaretheoremstyle[headpunct={},
numberwithin=section,
headfont=\normalfont\bfseries,
notefont=\bfseries, notebraces={}{},
postheadspace=\newline
%	preheadhook={\hspace{\parindent}}
]{mystyle}
\declaretheorem[style=mystyle,name=Th]{theorem}
\declaretheorem[style=mystyle,name=Lem]{lemma}
\declaretheorem[style=mystyle,name=Prop]{proposition}
\declaretheorem[style=mystyle,name=Def]{definition}
%\declaretheoremstyle[headpunct={},
%numberwithin=section,
%headfont=\normalfont\bfseries,
%notefont=\bfseries, notebraces={}{}
%]{mystyle2}
\declaretheorem[style=mystyle,name=Cor]{corollary}
\declaretheorem[style=mystyle,name=Eg]{example}

% 参考文献引用使用上标
\newcommand{\citep}[1]{\textsuperscript{\cite{#1}}}

% 自定义指令
\renewcommand{\leq}{\leqslant}
\renewcommand{\geq}{\geqslant}
\renewcommand{\vec}{\overrightarrow}
\renewcommand{\Re}{\mathrm{Re}}
\renewcommand{\Im}{\mathrm{Im}}
\renewcommand{\d}{\mathrm{d}}
\renewcommand{\i}{\mathrm{i}} 
\newcommand{\e}{\mathrm{e}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\F}{\mathbb{F}}
\newcommand{\Z}{\mathbb{Z}}
\renewcommand{\phi}{\varphi}
\newcommand{\eps}{\varepsilon}
\newcommand{\ept}{\varnothing}
\renewcommand{\hat}[1]{\widehat{#1}}
\renewcommand{\tilde}[1]{\widetilde{#1}}

\RequirePackage{subfiles}
