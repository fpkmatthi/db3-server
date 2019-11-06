" Title: plugins

" Plugins {{{
call plug#begin('~/.vim/plugged')

Plug 'airblade/vim-gitgutter'
" Plug 'ajh17/VimCompletesMe'
Plug 'easymotion/vim-easymotion'
Plug 'elzr/vim-json'
Plug 'itchyny/lightline.vim'
Plug 'jalvesaq/Nvim-R'
Plug 'jiangmiao/auto-pairs'
Plug 'junegunn/fzf.vim'
Plug 'junegunn/goyo.vim'
Plug 'junegunn/limelight.vim'
Plug 'junegunn/vim-easy-align'
Plug 'kovetskiy/sxhkd-vim'
" Plug 'lervag/vimtex'
Plug 'ludovicchabant/vim-gutentags'
Plug 'neovimhaskell/haskell-vim'
Plug 'nicholaides/words-to-avoid.vim'
Plug 'NLKNguyen/papercolor-theme'
Plug 'tpope/vim-commentary'
Plug 'tpope/vim-fugitive'
Plug 'tpope/vim-surround'
Plug 'tpope/vim-eunuch'
" Plug 'maximbaz/lightline-ale'
" Plug 'maximbaz/lightline-trailing-whitespace'
Plug 'mhinz/vim-startify'
Plug 'OmniSharp/omnisharp-vim'
Plug 'pangloss/vim-javascript'
" Plug 'plasticboy/vim-markdown'
" Plug 'PotatoesMaster/i3-vim-syntax'
Plug 'PProvost/vim-ps1'
Plug 'scrooloose/nerdtree'
" Plug 'SirVer/ultisnips'
" Plug 'vim-pandoc/vim-pandoc'
" Plug 'vim-pandoc/vim-pandoc-syntax'
Plug 'vim-scripts/indentpython.vim'
Plug 'vimwiki/vimwiki'
" Plug 'vim-syntastic/syntastic'
Plug 'w0rp/ale'
Plug 'xuhdev/vim-latex-live-preview', { 'for': 'tex' }
Plug 'ycm-core/YouCompleteMe'

call plug#end()
" }}}

" Plugin Options {{{
" Remap leader {{{
let mapleader = ","
" }}}

" Git Gutter {{{
let g:gitgutter_enabled = 1
let g:gitgutter_eager = 0
let g:gitgutter_max_signs = 500  " default value
let g:gitgutter_highlight_lines = 0
let g:gitgutter_override_sign_column_highlight = 0
" gutter symbols
let g:gitgutter_sign_removed = '-'
" TODO: edit gutter colours
" highlight GitGutterAdd ctermfg=2 ctermbg=0
" highlight GitGutterChange ctermfg=3 ctermbg=0
" highlight GitGutterDelete ctermfg=1 ctermbg=0

" No mappings set by gitgutter with let g:gitgutter_map_keys = 0
nmap <leader>nh <Plug>GitGutterNextHunk
nmap <leader>Nh <Plug>GitGutterPrevHunk
nmap <Leader>gsh <Plug>GitGutterStageHunk
nmap <Leader>guh <Plug>GitGutterUndoHunk
nmap <Leader>gph <Plug>GitGutterPreviewHunk
omap ih <Plug>GitGutterTextObjectInnerPending
omap ah <Plug>GitGutterTextObjectOuterPending
xmap ih <Plug>GitGutterTextObjectInnerVisual
xmap ah <Plug>GitGutterTextObjectOuterVisual
" }}}

" Lightline Options {{{
let g:lightline = {
    \ 'colorscheme': 'PaperColor',
    \ 'separator': { 'left': '', 'right': '' },
    \ 'subseparator': { 'left': '', 'right': '' }
\ }
" }}}

" Color scheme Options {{{
highlight clear SignColumn
set background=dark
colorscheme PaperColor
let g:PaperColor_Theme_Options = {
\   'theme': {
\     'default': {
\       'transparent_background': 1
\     }
\   }
\ }
" Custom function to toggle between light and dark background to keep the
" CursorLine settings since the default way of the solarized plugin also
" changes the CursorLine settings
function Togglebackground()
    let &background = ( &background == "dark"? "light" : "dark" ) 
    highlight LineNr ctermbg=None
    highlight CursorLine ctermbg=None
    highlight CursorLine cterm=Underline
    highlight CursorLineNr ctermbg=None
endfunction
map <F5> :call Togglebackground()<cr>
" }}}

" Easy-motion {{{
let g:EasyMotion_startofline = 0 " keep cursor column when JK motion
let g:EasyMotion_smartcase = 1

nmap s <Plug>(easymotion-overwin-f)
" Avoid repetitive use of the h j k l keys.
map <leader><leader>l <Plug>(easymotion-lineforward)
map <leader><leader>j <Plug>(easymotion-j)
map <leader><leader>k <Plug>(easymotion-k)
map <leader><leader>h <Plug>(easymotion-linebackward)
" replace default forward slash searching
map  / <Plug>(easymotion-sn)
omap / <Plug>(easymotion-tn)
map  n <Plug>(easymotion-next)
map  N <Plug>(easymotion-prev)
" }}}

" Nvim-R Options {{{
let R_openpdf = 1
" }}}

" Vim-gutentags Options {{{
set statusline+=%{gutentags#statusline()}
let g:gutentags_project_root = [".git","Vagrantfile","Makefile"]
let g:gutentags_ctags_tagfile = '.tags'
let g:gutentags_cache_dir = "~/.ctags/project-tags"
" let g:gutentags_enabled = 1
" }}}

" Vim-commentary Options {{{
" To add an unsupported filetype: autocmd FileType apache setlocal commentstring=#\ %s
" }}}

" Vim-surround Options {{{

" }}}

" Omnisharp Options {{{
let g:OmniSharp_server_stdio = 1
" }}}

" Vimwiki Options {{{
set nocompatible
filetype plugin on
" nmap <leader><leader>vwnl <Plug>VimwikiNextLink
let g:vimwiki_list = [{'path': '~/vimwiki/', 'syntax': 'markdown', 'ext': '.md'}]
" }}}

" FZF Options {{{
let g:fzf_action = {
  \ 'ctrl-t': 'tab split',
  \ 'ctrl-h': 'split',
  \ 'ctrl-v': 'vsplit'
\ }
" }}}

" Goyo Options {{{

" }}}

" Limelight Options {{{
" " Color name (:help cterm-colors) or ANSI code
let g:limelight_conceal_ctermfg = 'gray'
let g:limelight_conceal_ctermfg = 240
" }}}

" Wal Options {{{
" colorscheme wal
" }}}

" Vim-latex-live-preview Options {{{
map <Leader>c :LLPStartPreview<cr>
let g:livepreview_previewer = "zathura"
" let g:livepreview_cursorhold_recompile = 0
" }}}

" Easy-align Options {{{
" Start interactive EasyAlign in visual mode (e.g. vipga)
xmap ga <Plug>(EasyAlign)

" Start interactive EasyAlign for a motion/text object (e.g. gaip)
nmap ga <Plug>(EasyAlign)
" }}}

" YouCompleteMe Options {{{
let g:loaded_python_provider = 0
let g:python3_host_prog = '/bin/python3'
let g:ycm_autoclose_preview_window_after_completion=1
map <leader>g  :YcmCompleter GoToDefinitionElseDeclaration<CR>
" " python with virtualenv support
" py << EOF
" import os
" import sys
" if 'VIRTUAL_ENV' in os.environ:
"   project_base_dir = os.environ['VIRTUAL_ENV']
"   activate_this = os.path.join(project_base_dir, 'bin/activate_this.py')
"   execfile(activate_this, dict(__file__=activate_this))
" EOF
" }}}
" }}}
