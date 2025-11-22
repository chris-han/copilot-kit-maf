import { Folder as LucideFolder, LucideProps } from 'lucide-react';

const Folder = ({ className, ...props }: LucideProps) => {
  return <LucideFolder className={className} {...props} />;
};

export default Folder;