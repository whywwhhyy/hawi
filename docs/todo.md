- [ ] 重写agent为多队列单循环：
    - [ ] 1. 插队消息队列
    - [ ] 2. 工具调用请求队列
    - [ ] 3. 用户消息队列

- [ ] agent messages提供标签和id功能，以支持插件和工具实现便捷的上下文管理方法

- [ ] 还需要提供持久化机制，确保agent可以做到不管什么时候crash再拉起来都可以恢复session

- [ ] agent打断接口
    - [x] 添加打断接口
    - [ ] 实现agent打断接口

- [ ] 增加retry事件

- [x] Agent改名为Bao

- [ ] 给plugin加上name和tags字段，设计依赖管理和冲突识别的机制

- [ ] 将skills plugin拆分为terminal和skills

- [ ] 重构渐进式加载的tool设计，新增brief字段，使用tool_help来获得tool的详细介绍，run_tool来执行tool

- [x] 优化各个hook的参数

- [ ] tool result导致爆上下文的时候，撤销tool result并且提示模型

- [ ] “需要用户确认”这个动作要做成一个hook，并且仅支持一个hook

- [ ] https://zhuanlan.zhihu.com/p/1943399204027373513?share_code=1roZMkugobZJr&utm_psn=2020735776124667643