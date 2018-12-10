import re
from PIL import Image

with open('/data5/xin/irv2_atrous/train.txt', 'wa+') as g:
  with open('/data5/xin/irv2_atrous/all.txt', 'rb') as f:
    for line in f.readlines():
      path,width,height,left,top,right,bottom,label = line.rstrip().split(',')
      path = path.replace('ryan', 'xin')
      g.write(','.join([path,width,height,left,top,right,bottom,label]) + '\n')

label_map = {}
with open('/data5/xin/irv2_atrous/logos', 'rb') as f:
  for line in f.readlines():
    label = line.strip().split()[-1].strip()
    label_map[re.sub('[-_]','',label)] = label
    label_map[label] = label
label_map['petrobas'] = 'petrobras'
label_map['castrol'] = 'castrol_oil_company'
label_map['bt'] = 'bt_sport'
label_map['santander'] = 'santander_bank'
label_map['epson'] = 'epson_america_inc'
label_map['mercedesamg'] = 'mercedes-benz'
label_map['hiltonhotels'] = 'hilton'

with open('/data5/xin/irv2_atrous/test.txt', 'wa+') as g:
  with open('/data5/xin/irv2_atrous/hive_test.bboxes', 'rb') as f:
    for line in f.readlines():
      path,width,height,left,top,right,bottom,label = line.rstrip().split(',')
      path = path.replace('ryan', 'xin')
      image = Image.open(path)
      width, height = image.size
      try:
        label = label_map[label]
      except:
        print('>>>>>>>>>>>', label, line)
      g.write(','.join([path,str(width),str(height),left,top,right,bottom,label]) + '\n')
