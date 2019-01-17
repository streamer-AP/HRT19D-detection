from __future__ import print_function

import roslib; roslib.load_manifest('husky_control')
import rospy

from geometry_msgs.msg import Twist

import sys, select, termios, tty


#the format of  
msg = """
Reading from the keyboard  and Publishing to Twist!
---------------------------
Moving around:
   u    i    o
   j  
"""


### the key and value
moveBindings = {
        'i':(1,0,0,0),
        'o':(1,0,0,-1),
        'j':(0,0,0,1),
        'u':(1,0,0,1),
    }

#deal with the information from the keyboard
def getKey():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def vels(speed,turn):
    return "currently:\tspeed %s\tturn %s " % (speed,turn)

if __name__=="__main__":
    settings = termios.tcgetattr(sys.stdin)

    pub = rospy.Publisher('cmd_vel', Twist, queue_size = 1) #the topic is cmd_vel,the Twist is message 
    rospy.init_node('husky_control')#initialize the node

    speed = rospy.get_param("~speed", 0.5)
    turn = rospy.get_param("~turn", 1.0)
    x = 0
    y = 0
    z = 0
    th = 0
    status = 0

    try:
        print(msg)
        print(vels(speed,turn))
        while(1):
            key = getKey()
            if key in moveBindings.keys():#use the value from key
                x = moveBindings[key][0]
                y = moveBindings[key][1]
                z = moveBindings[key][2]
                th = moveBindings[key][3]

                print(vels(speed,turn))
                    print(msg) #send the msg to control console
            else:#all are set to NULL when there is no inputs
                x = 0
                y = 0
                z = 0
                th = 0

            twist = Twist()
            twist.linear.x = x*speed; twist.linear.y = y*speed; twist.linear.z = z*speed#get the linear from x,y,z
            twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = th*turn#get the angle 
            pub.publish(twist)

    except Exception as e:
        print(e)

    finally:
        twist = Twist()
        twist.linear.x = 0; twist.linear.y = 0; twist.linear.z = 0
        twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0
        pub.publish(twist)#send the msg to topic

termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)