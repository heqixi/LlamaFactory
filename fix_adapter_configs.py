"""
修复从服务器拷贝的 LoRA 模型的 adapter_config.json 中的 base_model_name_or_path
"""
import json
import os
from pathlib import Path

# 路径映射表
PATH_MAPPING = {
    "/root/.cache/modelscope/hub/models/Qwen/Qwen2-0___5B-Instruct": "Qwen/Qwen2-0.5B-Instruct",
    "/root/.cache/modelscope/hub/models/Qwen/Qwen2-1___5B-Instruct": "Qwen/Qwen2-1.5B-Instruct",
    "/root/.cache/modelscope/hub/models/Qwen/Qwen2-7B-Instruct": "Qwen/Qwen2-7B-Instruct",
    # 添加更多映射...
}

def fix_config(config_path: Path):
    """修复单个配置文件"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        old_path = config.get("base_model_name_or_path", "")
        
        # 检查是否需要修复
        if old_path.startswith("/") or "modelscope" in old_path:
            # 尝试从映射表中找到对应的模型名
            new_path = PATH_MAPPING.get(old_path)
            
            if not new_path:
                # 尝试从路径中推断模型名
                if "Qwen2-0" in old_path or "Qwen2-0.5B" in old_path or "Qwen2-0___5B" in old_path:
                    new_path = "Qwen/Qwen2-0.5B-Instruct"
                elif "Qwen2-1" in old_path or "Qwen2-1.5B" in old_path:
                    new_path = "Qwen/Qwen2-1.5B-Instruct"
                elif "Qwen2-7B" in old_path:
                    new_path = "Qwen/Qwen2-7B-Instruct"
                else:
                    print(f"  [!] 无法识别模型: {old_path}")
                    return False
            
            config["base_model_name_or_path"] = new_path
            
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"  [OK] {old_path} -> {new_path}")
            return True
        else:
            print(f"  [SKIP] 已是正确格式: {old_path}")
            return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

def main():
    saves_dir = Path(__file__).parent / "saves"
    
    if not saves_dir.exists():
        print(f"saves 目录不存在: {saves_dir}")
        return
    
    print("=" * 60)
    print("修复 LoRA adapter_config.json 中的 base_model_name_or_path")
    print("=" * 60)
    
    fixed_count = 0
    total_count = 0
    
    for config_path in saves_dir.rglob("adapter_config.json"):
        total_count += 1
        print(f"\n处理: {config_path.relative_to(saves_dir)}")
        if fix_config(config_path):
            fixed_count += 1
    
    print("\n" + "=" * 60)
    print(f"完成! 共处理 {total_count} 个文件，修复了 {fixed_count} 个")
    print("=" * 60)

if __name__ == "__main__":
    main()

