" General
augroup shared_mappings
  autocmd Filetype tex,rmd inoremap <Space><Space> <Esc>/<++><Enter>"_c4l
augroup END

" Latex
augroup tex_mappings
  autocmd FileType tex inoremap ;s \section{}<Enter><Enter><++><Esc>2k$F{a
  autocmd FileType tex inoremap ;ss \subsection{}<Enter><Enter><++><Esc>2k$F{a
  autocmd FileType tex inoremap ;sss \subsubsection{}<Enter><Enter><++><Esc>2k$F{a
  autocmd FileType tex inoremap ;be \begin{enumerate}<Enter>\item<Space><Enter>\end{enumerate}<Enter><++><Esc>2k$a
  autocmd FileType tex inoremap ;bi \begin{itemize}<Enter>\item<Space><Enter>\end{itemize}<Enter><++><Esc>2k$a
  autocmd FileType tex inoremap ;l \label{}<++><Esc>F{a
  autocmd FileType tex inoremap ;b \textbf{}<++><Esc>F{a
  autocmd FileType tex inoremap ;i \textit{}<++><Esc>F{a
  autocmd FileType tex inoremap ;sq `'<++><Esc>F`a
  autocmd FileType tex inoremap ;dq ``"<++><Esc>F`a
  autocmd FileType tex inoremap ;tc \textcite{}<++><Esc>F{a
  autocmd FileType tex inoremap ;pc \parencite{}<++><Esc>F{a
  autocmd FileType tex inoremap ;v \verb\|\|<++><Esc>ba
augroup END

" Rmarkdown
augroup rmd_mappings
  autocmd FileType rmd inoremap ;r ```{r}<Enter><++><Enter>```<Esc>2kfra
  autocmd FileType rmd noremap <F6> :w<Enter>:!echo<space>"require(rmarkdown);<Space>render('<c-r>%')"<Space>\|<Space>R<Space>--vanilla<Enter>
augroup END
