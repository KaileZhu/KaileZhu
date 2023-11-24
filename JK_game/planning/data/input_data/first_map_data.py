#map   node position and place
#area name represent a polynorm list
x1=[(78+i)*111*0.855 for i in [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]]
x2=[(78+i)*111*0.855 for i in [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]]
x3=[(79+i)*111*0.855 for i in [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]]
x4=[(79+i)*111*0.855 for i in [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]]

y1=[(29+i)*111 for i in [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]]
y2=[(30+i)*111 for i in [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]]
y3=[(30+i)*111 for i in [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]]
y4=[(29+i)*111 for i in [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]]
position={}
name_list={0:'a',1:'b',2:'c',3:'d',4:'e',5:'f',6:'g',7:'h',8:'i',9:'j',10:'k',11:'l',12:'m',13:'n',14:'o',15:'p',16:'q',17:'r',18:'s',19:'t',20:'u',21:'v',22:'w',
23:'x',24:'y',25:'z'}
position_center={}
for i in range(16):
    position[name_list[i]]=[[x1[i],y1[i]],[x2[i],y2[i]],
                            [x3[i],y3[i]],[x4[i],y4[i]]]
    position_center[name_list[i]]=[(x1[i]+x2[i]+x3[i]+x4[i])/4,(y1[i]+y2[i]+y3[i]+y4[i])/4]
print(position)
#这里的position有问题，定义上不合适，需要改成新的问题
#输入值为
#position['l']=[[7497.495-150, 3330-150], [7497.495-150, 3441+150], [7592.4+150, 3441+150], [7592.4+150, 3330-150]]