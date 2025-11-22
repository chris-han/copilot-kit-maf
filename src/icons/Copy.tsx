import { Copy as LucideCopy, LucideProps } from 'lucide-react';

const Copy = ({ className, ...props }: LucideProps) => {
  return <LucideCopy className={className} {...props} />;
};

export default Copy;