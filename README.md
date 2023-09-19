## Dependencies
sudo apt-get install libyaml-cpp-dev

## input 
/livox/lidar_192_168_147_231      : sensor_msgs/PointCloud2
/livox/lidar_192_168_147_232      : sensor_msgs/PointCloud2
/livox/lidar_192_168_147_233      : sensor_msgs/PointCloud2


## coding

```
for (int i = 0; i < (int) lidar0->fields.size(); ++i) 
{
    std::cout << "lidar0: " << lidar0->fields[i].name << std::endl;
}
```
lidar0: intensity
lidar0: x
lidar0: y
lidar0: z