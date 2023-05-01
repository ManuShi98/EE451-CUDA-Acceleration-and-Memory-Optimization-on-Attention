import matplotlib.pyplot as plt
import numpy as np

forward_time_flash = [63.73504027724267, 60.021760649979115, 59.501120634377, 60.54688058793545, 60.3366407006979, 59.36224054545164, 61.89824055880308, 59.45056065917015, 60.97216069698334, 59.90976069122553, 76.5760025382042, 76.85440123081207, 77.10592232644558, 76.52768202126026, 78.30495797097683, 81.58175826072693]
forward_time_block = [327.90398597717285, 112.5119999051094, 95.0080007314682, 203.77600193023682, 97.28000313043594, 93.02400052547455, 227.32800245285034, 101.27999633550644, 92.19200164079666, 187.6479983329773, 102.36799716949463, 94.01600062847137, 187.55200505256653, 94.46399658918381, 195.5839991569519, 93.31200271844864]
backward_time = []


batch_size = [i for i in range(1, 17)]


plt.plot(batch_size, forward_time_flash, label='flash')
plt.plot(batch_size, forward_time_block, label='block')
plt.xlabel('Batch Size')
plt.ylabel('Time (ms)')
ax = plt.gca()
ax.yaxis.set_label_coords(-0.12, 0.5)
plt.legend()
plt.grid(True)
plt.title('Forward Time')
plt.savefig('forward_time_batch_size.png')

plt.clf()

forward_time_flash = [63.67552045732737, 60.60384061187506, 60.353920459747314, 61.04640066623688, 62.69792079925537, 59.948800802230835, 64.0044804662466, 60.34592047333717, 61.649600602686405, 60.09024050086737, 76.38464257121086, 76.43392220139503, 77.34368167817593, 76.9523212313652, 78.1263979524374, 81.4595178514719]
forward_time_block = [408.28800201416016, 117.66400188207626, 102.65599936246872, 99.48799759149551, 97.28000313043594, 111.87200248241425, 208.8959962129593, 109.56799983978271, 94.97600048780441, 92.96000003814697, 97.08800166845322, 220.0320065021515, 97.37599641084671, 94.2080020904541, 104.44799810647964, 94.2080020904541]

head_num = [i for i in range(1, 17)]


plt.plot(head_num, forward_time_flash, label='flash')
plt.plot(head_num, forward_time_block, label='block')
plt.xlabel('Head Number')
plt.ylabel('Time (ms)')
ax = plt.gca()
ax.yaxis.set_label_coords(-0.12, 0.5)
plt.legend()
plt.grid(True)
plt.title('Forward Time')
plt.savefig('forward_time_head_num.png')

plt.clf()

forward_time_flash = [62.62176051735878, 77.80352219939232, 191.41664013266563, 633.9033526182175]
forward_time_block = [328.70399951934814, 235.6799989938736, 225.2800017595291, 402.17599272727966]
token_num = [i for i in range(5, 9)]


plt.plot(token_num, forward_time_flash, label='flash')
plt.plot(token_num, forward_time_block, label='block')
plt.xlabel('Token Number')
plt.ylabel('Time (ms)')
plt.xticks([i for i in range(5, 9)], [2**i for i in range(5, 9)])
ax = plt.gca()
ax.yaxis.set_label_coords(-0.12, 0.5)
plt.legend()
plt.grid(True)
plt.title('Forward Time')
plt.savefig('forward_time_token_num.png')

plt.clf()


# plt.plot(batch_size, backward_time, label='Block Size = 1')
# plt.xlabel('Batch Size')
# plt.ylabel('Time (ms)')
# ax = plt.gca()
# ax.yaxis.set_label_coords(-0.12, 0.5)
# plt.legend()
# plt.grid(True)
# plt.title('Backward Time')
# plt.savefig('backward_time.png')