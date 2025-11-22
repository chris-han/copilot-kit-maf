import { File as LucideFile, LucideProps } from 'lucide-react';

const File = ({ className, ...props }: LucideProps) => {
  return <LucideFile className={className} {...props} />;
};

export default File;