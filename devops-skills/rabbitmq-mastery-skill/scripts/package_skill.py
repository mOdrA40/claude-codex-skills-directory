#!/usr/bin/env python3
"""
RabbitMQ Skill Packager
Packages the skill directory into a distributable .skill file (zip format)
"""

import os
import zipfile
import hashlib
import json
from datetime import datetime
from pathlib import Path


def calculate_checksum(filepath: str) -> str:
    """Calculate SHA256 checksum of a file"""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_file_stats(skill_dir: Path) -> dict:
    """Gather statistics about skill files"""
    stats = {
        'total_files': 0,
        'total_size': 0,
        'files': [],
        'categories': {
            'markdown': {'count': 0, 'size': 0},
            'python': {'count': 0, 'size': 0},
            'other': {'count': 0, 'size': 0}
        }
    }
    
    for root, dirs, files in os.walk(skill_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file.startswith('.'):
                continue
                
            filepath = Path(root) / file
            size = filepath.stat().st_size
            rel_path = filepath.relative_to(skill_dir)
            
            stats['total_files'] += 1
            stats['total_size'] += size
            stats['files'].append({
                'path': str(rel_path),
                'size': size,
                'checksum': calculate_checksum(str(filepath))
            })
            
            # Categorize
            ext = filepath.suffix.lower()
            if ext == '.md':
                stats['categories']['markdown']['count'] += 1
                stats['categories']['markdown']['size'] += size
            elif ext == '.py':
                stats['categories']['python']['count'] += 1
                stats['categories']['python']['size'] += size
            else:
                stats['categories']['other']['count'] += 1
                stats['categories']['other']['size'] += size
    
    return stats


def package_skill(skill_dir: str, output_dir: str = None) -> str:
    """
    Package a skill directory into a .skill file
    
    Args:
        skill_dir: Path to skill directory containing SKILL.md
        output_dir: Output directory (defaults to current directory)
    
    Returns:
        Path to created .skill file
    """
    skill_path = Path(skill_dir).resolve()
    
    # Validate skill directory
    skill_md = skill_path / 'SKILL.md'
    if not skill_md.exists():
        raise FileNotFoundError(f"SKILL.md not found in {skill_path}")
    
    # Extract skill name from directory
    skill_name = skill_path.name
    
    # Create output path
    if output_dir:
        output_path = Path(output_dir).resolve()
    else:
        output_path = Path.cwd()
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate version timestamp
    version = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_path / f"{skill_name}-{version}.skill"
    
    # Gather file statistics
    print(f"📦 Packaging skill: {skill_name}")
    print(f"   Source: {skill_path}")
    stats = get_file_stats(skill_path)
    
    # Create manifest
    manifest = {
        'skill_name': skill_name,
        'version': version,
        'created_at': datetime.now().isoformat(),
        'description': 'RabbitMQ Master Skill - Production-grade messaging expertise',
        'author': 'Claude AI',
        'statistics': {
            'total_files': stats['total_files'],
            'total_size_bytes': stats['total_size'],
            'categories': stats['categories']
        },
        'files': stats['files'],
        'entry_point': 'SKILL.md',
        'required_tools': ['bash', 'python3'],
        'tags': [
            'rabbitmq', 'messaging', 'distributed-systems', 
            'high-availability', 'performance', 'security'
        ]
    }
    
    # Create zip file with .skill extension
    print(f"   Creating package...")
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add manifest first
        manifest_json = json.dumps(manifest, indent=2)
        zf.writestr('MANIFEST.json', manifest_json)
        
        # Add all skill files
        for root, dirs, files in os.walk(skill_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.startswith('.'):
                    continue
                    
                filepath = Path(root) / file
                arcname = filepath.relative_to(skill_path)
                zf.write(filepath, arcname)
    
    # Calculate package checksum
    package_checksum = calculate_checksum(str(output_file))
    package_size = output_file.stat().st_size
    
    # Print summary
    print(f"\n✅ Skill packaged successfully!")
    print(f"   Output: {output_file}")
    print(f"   Size: {package_size:,} bytes ({package_size/1024:.1f} KB)")
    print(f"   Checksum (SHA256): {package_checksum}")
    print(f"\n📊 Contents Summary:")
    print(f"   Total files: {stats['total_files']}")
    print(f"   Markdown docs: {stats['categories']['markdown']['count']} files ({stats['categories']['markdown']['size']/1024:.1f} KB)")
    print(f"   Python scripts: {stats['categories']['python']['count']} files ({stats['categories']['python']['size']/1024:.1f} KB)")
    
    return str(output_file)


def verify_skill(skill_file: str) -> bool:
    """Verify a .skill package integrity"""
    skill_path = Path(skill_file)
    
    if not skill_path.exists():
        print(f"❌ File not found: {skill_path}")
        return False
    
    print(f"🔍 Verifying: {skill_path.name}")
    
    try:
        with zipfile.ZipFile(skill_path, 'r') as zf:
            # Check for manifest
            if 'MANIFEST.json' not in zf.namelist():
                print("❌ Missing MANIFEST.json")
                return False
            
            # Load and validate manifest
            manifest = json.loads(zf.read('MANIFEST.json'))
            
            # Check entry point exists
            entry_point = manifest.get('entry_point', 'SKILL.md')
            if entry_point not in zf.namelist():
                print(f"❌ Missing entry point: {entry_point}")
                return False
            
            # Verify all files in manifest exist
            manifest_files = {f['path'] for f in manifest.get('files', [])}
            archive_files = {n for n in zf.namelist() if n != 'MANIFEST.json'}
            
            missing = manifest_files - archive_files
            if missing:
                print(f"❌ Missing files: {missing}")
                return False
            
            # Test archive integrity
            bad_file = zf.testzip()
            if bad_file:
                print(f"❌ Corrupted file: {bad_file}")
                return False
            
            print(f"✅ Package verified successfully!")
            print(f"   Skill: {manifest['skill_name']}")
            print(f"   Version: {manifest['version']}")
            print(f"   Files: {manifest['statistics']['total_files']}")
            return True
            
    except zipfile.BadZipFile:
        print("❌ Invalid zip file")
        return False
    except json.JSONDecodeError:
        print("❌ Invalid manifest JSON")
        return False


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Package: python package_skill.py <skill_directory> [output_directory]")
        print("  Verify:  python package_skill.py --verify <skill_file>")
        sys.exit(1)
    
    if sys.argv[1] == '--verify':
        if len(sys.argv) < 3:
            print("Error: Please specify skill file to verify")
            sys.exit(1)
        success = verify_skill(sys.argv[2])
        sys.exit(0 if success else 1)
    else:
        skill_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else '/mnt/user-data/outputs'
        
        try:
            output_file = package_skill(skill_dir, output_dir)
            print(f"\n🎉 Done! Skill ready for distribution.")
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
