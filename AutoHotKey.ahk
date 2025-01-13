; Typora
; 快捷增加字体颜色
; SendInput {Text} 解决中文输入法问题
 
#IfWinActive ahk_exe Typora.exe
{
    ; Ctrl+Alt+O 橙色
    ^!o::addFontColor("orange")
 
    ; Ctrl+Alt+R 红色
    ^!r::addFontColor("red")
 
    ; Ctrl+Alt+Y 黄色
    ^!y::addFontColor("yellow")

    ; Ctrl+Alt+R 灰色
    ^!g::addFontColor("grey")

    ; Ctrl+Alt+B 浅蓝色
    ^!b::addFontColor("cornflowerblue")
}
 
; 快捷增加字体颜色
addFontColor(color){
    clipboard := "" ; 清空剪切板
    Send {ctrl down}c{ctrl up} ; 复制
    SendInput {TEXT}<font color='%color%'>
    SendInput {ctrl down}v{ctrl up} ; 粘贴
    If(clipboard = ""){
        SendInput {TEXT}</font> ; Typora 在这不会自动补充
    }else{
        SendInput {TEXT}</ ; Typora中自动补全标签
    }
}
